import numpy as np
import numpy.polynomial.chebyshev as ch
import numpy.polynomial.chebyshev as ch
from numpy.polynomial.chebyshev import Chebyshev as ChebyshevNP
from numba import njit
import numba as nb

@njit(cache=True)
def cheby_D_jit(n, dtype = np.float64):
    Dmat = np.zeros((n, n), dtype=dtype)
    j = 1
    while j < n:
        Dmat[0,j] = j
        j += 2
    for i in range(1,n):
        j = i + 1
        while j < n:
            Dmat[i,j] = 2*j
            j += 2
    return Dmat

# @njit([nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:](nb.complex128[:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:](nb.float64[:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:](nb.complex128[:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_evaluation(coeffs, domain, x):
    # x = 0.5*((domain[1] - domain[0])*x_shift + domain[1] + domain[0])
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp1 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp0 = coeffs[N - 1]*(np.zeros(x.shape) + 1)
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:,:](nb.float64[:,:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_evaluation_2d(coeffs, domain, x):
    # x = 0.5*((domain[1] - domain[0])*x_shift + domain[1] + domain[0])
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    
    N = coeffs.shape[0]
    outshape = coeffs.shape + x.shape

    bkp2 = np.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp1 = np.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*np.ones(outshape[1:])
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64[:](nb.float64[:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:](nb.complex128[:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:](nb.float64[:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:](nb.complex128[:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_deriv_evaluation(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp1 = np.zeros(x.shape, dtype = type(coeffs[0]))
    bkp0 = coeffs[N - 1]*(np.zeros(x.shape) + 1)
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

# @njit([nb.float64[:,:](nb.float64[:,:], nb.float64[:], nb.float64[:]),
#        nb.complex128[:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:]),
#        nb.float64[:,:,:](nb.float64[:,:], nb.float64[:], nb.float64[:,:]),
#        nb.complex128[:,:,:](nb.complex128[:,:], nb.float64[:], nb.float64[:,:])], cache=True)
def clenshaw_deriv_evaluation_2d(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])
    
    N = coeffs.shape[0]
    outshape = coeffs.shape + x.shape

    bkp2 = np.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp1 = np.zeros(outshape[1:], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*np.ones(outshape[1:])
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

# @njit([nb.float64(nb.float64[:], nb.float64[:], nb.float64),
#        nb.complex128(nb.complex128[:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_evaluation_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = 0.*coeffs[N - 1]
    bkp1 = 0.*coeffs[N - 1]
    bkp0 = coeffs[N - 1]
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64[:](nb.float64[:,:], nb.float64[:], nb.float64),
#        nb.complex128[:](nb.complex128[:,:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_evaluation_2d_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])

    N = coeffs.shape[0]

    bkp2 = np.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp1 = np.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*np.ones(coeffs.shape[1])
    for i in range(N - 2, 0, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        bkp0 = coeffs[i] + 2.*x_shift*bkp1 - bkp2

    return (coeffs[0] + x_shift*bkp0 - bkp1)

# @njit([nb.float64(nb.float64[:], nb.float64[:], nb.float64),
#        nb.complex128(nb.complex128[:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_deriv_evaluation_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])
    N = len(coeffs)

    bkp2 = 0.*coeffs[N - 1]
    bkp1 = 0.*coeffs[N - 1]
    bkp0 = coeffs[N - 1]
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

# @njit([nb.float64[:](nb.float64[:,:], nb.float64[:], nb.float64),
#        nb.complex128[:](nb.complex128[:,:], nb.float64[:], nb.float64)], cache=True)
def clenshaw_deriv_evaluation_2d_scalar(coeffs, domain, x):
    x_shift = (2.*x - (domain[1] + domain[0]))/(domain[1] - domain[0])
    dx = 2./(domain[1] - domain[0])

    N = coeffs.shape[0]

    bkp2 = np.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp1 = np.zeros(coeffs.shape[1], dtype = type(coeffs[0,0]))
    bkp0 = coeffs[N - 1]*np.ones(coeffs.shape[1])
    for i in range(N - 2, 1, -1):
        bkp2 = bkp1
        bkp1 = bkp0
        ii = nb.float64(i)
        alpha = 2.*x_shift*(ii + 1.)/(ii)
        beta = -(ii + 2.)/(ii)
        bkp0 = coeffs[i] + alpha*bkp1 + beta*bkp2

    i = 1
    beta = -nb.float64(i + 2.)/nb.float64(i)
    return dx*(coeffs[1] + 4*x_shift*bkp0 + beta*bkp1)

class Chebyshev(ChebyshevNP):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        atol = 1e-10
        diff0 = np.abs(1 - (atol + np.dot(np.ones(len(self.coef)), self.coef))/(atol + np.dot(np.ones(len(self.coef) - 1), self.coef[:-1])))
        self.error = diff0

    def custom_eval(self, arg):
        if isinstance(arg, np.ndarray):
            return clenshaw_evaluation(self.coef, self.domain, np.atleast_1d(arg))
        else:
            return clenshaw_evaluation_scalar(self.coef, self.domain, arg)
    
    def custom_deriv(self, arg):
        if isinstance(arg, np.ndarray):
            return clenshaw_deriv_evaluation(self.coef, self.domain, np.atleast_1d(arg))
        else:
            return clenshaw_deriv_evaluation_scalar(self.coef, self.domain, arg)
    
    def __call__(self, arg, deriv = 0):
        if deriv == 0:
            return self.custom_eval(arg)
        elif deriv == 1:
            return self.custom_deriv(arg)
        else:
            return super().deriv(deriv)(arg)
        
    @property
    def coeffs(self):
        return self.coef
    
def chebcolloc_1d_points(n):
    return ch.chebpts1(n)

def chebcolloc_1d_matrix_and_inverse(n):
    nodes = ch.chebpts1(n)
    Tmat0 = np.array(ch.chebvander(nodes, n-1))
    Tmat0Inv = np.linalg.inv(Tmat0)
    return Tmat0, Tmat0Inv

def chebcolloc_1d_matrix(n):
    nodes = ch.chebpts1(n)
    Tmat0 = np.array(ch.chebvander(nodes, n-1))
    return Tmat0
    
def chebcolloc_1d_coeffs(data):
    n = len(data)
    Tmat0 = chebcolloc_1d_matrix(n)
    coeffs = np.linalg.solve(Tmat0, data)

    return coeffs

def chebcolloc_1d(data, domain = [-1, 1]):
    coeffs = chebcolloc_1d_coeffs(data)
    return Chebyshev(coeffs, domain = domain)

from numpy.polynomial.chebyshev import chebval2d, chebder

class Chebyshev2D(ChebyshevNP):
    def __init__(self, coeffs, domains = [[-1, 1], [-1, 1]]):
        self.coef = coeffs
        self.domains = domains
        atol = 1e-10
        diff0 = np.abs(1 - (atol + np.dot(np.ones(len(self.coef)), self.coef))/(atol + np.dot(np.ones(len(self.coef) - 1), self.coef[:-1])))
        self.error = diff0
        self.dx = 2./(self.domains[0][1] - self.domains[0][0])
        self.dy = 2./(self.domains[1][1] - self.domains[1][0])

    def transform_args(self, x, y):
        x = (2.*x - (self.domains[0][1] + self.domains[0][0]))/(self.domains[0][1] - self.domains[0][0])
        y = (2.*y - (self.domains[1][1] + self.domains[1][0]))/(self.domains[1][1] - self.domains[1][0])
        return x, y

    def eval(self, x, y):
        x, y = self.transform_args(x, y)
        return chebval2d(x, y, self.coef)
    
    def deriv_x(self, x, y):
        x, y = self.transform_args(x, y)
        dc = chebder(self.coef, m = 1, scl = self.dx, axis = 0)
        return chebval2d(x, y, dc)
    
    def deriv_xx(self, x, y):
        x, y = self.transform_args(x, y)
        dc = chebder(self.coef, m = 2, scl = self.dx, axis = 0)
        return chebval2d(x, y, dc)

    def deriv_y(self, x, y):
        x, y = self.transform_args(x, y)
        dc = chebder(self.coef, m = 1, scl = self.dy, axis = 1)
        return chebval2d(x, y, dc)
    
    def deriv_yy(self, x, y):
        x, y = self.transform_args(x, y)
        dc = chebder(self.coef, m = 2, scl = self.dy, axis = 1)
        return chebval2d(x, y, dc)
    
    def deriv_yy(self, x, y):
        x, y = self.transform_args(x, y)
        dc = chebder(self.coef, m = 1, scl = self.dy, axis = 1)
        dc = chebder(dc, m = 1, scl = self.dx, axis = 0)
        return chebval2d(x, y, dc)
    
    def __call__(self, x, y):
        return self.eval(x, y)
        
    @property
    def coeffs(self):
        return self.coef
    
def chebcolloc_2d_coeffs(data):
    return chebcolloc_2d_coeffs_lazy(data)

def chebcolloc_2d_coeffs_lazy(data):
    n, m = data.shape
    return np.linalg.solve(chebcolloc_1d_matrix(n),[np.linalg.solve(chebcolloc_1d_matrix(m), x) for x in data])

def chebcolloc_2d(data, domains = [[-1, 1], [-1, 1]]):
    coeffs = chebcolloc_2d_coeffs(data)
    return Chebyshev2D(coeffs, domains = domains)