/* Private function declarations for libm.
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#define __MSUF_X(x, suffix) x ## suffix
#define __MSUF_S(...) __MSUF_X (__VA_ARGS__)
#define __MSUF(x) __MSUF_S (x, _MSUF_)

#define __MSUF_R_X(x, suffix) x ## suffix ## _r
#define __MSUF_R_S(...) __MSUF_R_X (__VA_ARGS__)
#define __MSUF_R(x) __MSUF_R_S (x, _MSUF_)

/* IEEE style elementary functions.  */
extern _Mdouble_ __MSUF (__ieee754_acos) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_acosh) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_asin) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_atan2) (_Mdouble_, _Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_atanh) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_cosh) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_exp) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_exp10) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_exp2) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_fmod) (_Mdouble_, _Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_gamma) (_Mdouble_);
extern _Mdouble_ __MSUF_R (__ieee754_gamma) (_Mdouble_, int *);
extern _Mdouble_ __MSUF (__ieee754_hypot) (_Mdouble_, _Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_j0) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_j1) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_jn) (int, _Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_lgamma) (_Mdouble_);
extern _Mdouble_ __MSUF_R (__ieee754_lgamma) (_Mdouble_, int *);
extern _Mdouble_ __MSUF (__ieee754_log) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_log10) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_log2) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_pow) (_Mdouble_, _Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_remainder) (_Mdouble_, _Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_sinh) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_sqrt) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_y0) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_y1) (_Mdouble_);
extern _Mdouble_ __MSUF (__ieee754_yn) (int, _Mdouble_);

extern _Mdouble_ __MSUF (__ieee754_scalb) (_Mdouble_, _Mdouble_);
extern int __MSUF (__ieee754_ilogb) (_Mdouble_);

extern int32_t __MSUF (__ieee754_rem_pio2) (_Mdouble_, _Mdouble_ *);

/* fdlibm kernel functions.  */
extern _Mdouble_ __MSUF (__kernel_sin) (_Mdouble_, _Mdouble_, int);
extern _Mdouble_ __MSUF (__kernel_cos) (_Mdouble_, _Mdouble_);
extern _Mdouble_ __MSUF (__kernel_tan) (_Mdouble_, _Mdouble_, int);

#if defined __MATH_DECLARING_LONG_DOUBLE || defined __MATH_DECLARING_FLOATN
extern void __MSUF (__kernel_sincos) (_Mdouble_, _Mdouble_,
				      _Mdouble_ *, _Mdouble_ *, int);
#endif

#if !defined __MATH_DECLARING_LONG_DOUBLE || defined __MATH_DECLARING_FLOATN
extern int __MSUF (__kernel_rem_pio2) (_Mdouble_ *, _Mdouble_ *, int,
				       int, int, const int32_t *);
#endif

/* Internal functions.  */

/* Return X^2 + Y^2 - 1, computed without large cancellation error.
   It is given that 1 > X >= Y >= epsilon / 2, and that X^2 + Y^2 >=
   0.5.  */
extern _Mdouble_ __MSUF (__x2y2m1) (_Mdouble_ x, _Mdouble_ y);

/* Compute the product of X + X_EPS, X + X_EPS + 1, ..., X + X_EPS + N
   - 1, in the form R * (1 + *EPS) where the return value R is an
   approximation to the product and *EPS is set to indicate the
   approximate error in the return value.  X is such that all the
   values X + 1, ..., X + N - 1 are exactly representable, and X_EPS /
   X is small enough that factors quadratic in it can be
   neglected.  */
extern _Mdouble_ __MSUF (__gamma_product) (_Mdouble_ x, _Mdouble_ x_eps,
					   int n, _Mdouble_ *eps);

/* Compute lgamma of a negative argument X, if it is in a range
   (depending on the floating-point format) for which expansion around
   zeros is used, setting *SIGNGAMP accordingly.  */
extern _Mdouble_ __MSUF (__lgamma_neg) (_Mdouble_ x, int *signgamp);

/* Compute the product of 1 + (T / (X + X_EPS)), 1 + (T / (X + X_EPS +
   1)), ..., 1 + (T / (X + X_EPS + N - 1)), minus 1.  X is such that
   all the values X + 1, ..., X + N - 1 are exactly representable, and
   X_EPS / X is small enough that factors quadratic in it can be
   neglected.  */
#if !defined __MATH_DECLARING_FLOAT
extern _Mdouble_ __MSUF (__lgamma_product) (_Mdouble_ t, _Mdouble_ x,
					    _Mdouble_ x_eps, int n);
#endif

#undef __MSUF_X
#undef __MSUF_S
#undef __MSUF

#undef __MSUF_R_X
#undef __MSUF_R_S
#undef __MSUF_R
