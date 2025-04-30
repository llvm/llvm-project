/* Implementation of gamma function according to ISO C.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <math.h>
#include <math-narrow-eval.h>
#include <math_private.h>
#include <fenv_private.h>
#include <math-underflow.h>
#include <float.h>
#include <libm-alias-finite.h>
#include <mul_split.h>

/* Coefficients B_2k / 2k(2k-1) of x^-(2k-1) inside exp in Stirling's
   approximation to gamma function.  */

static const double gamma_coeff[] =
  {
    0x1.5555555555555p-4,
    -0xb.60b60b60b60b8p-12,
    0x3.4034034034034p-12,
    -0x2.7027027027028p-12,
    0x3.72a3c5631fe46p-12,
    -0x7.daac36664f1f4p-12,
  };

#define NCOEFF (sizeof (gamma_coeff) / sizeof (gamma_coeff[0]))

/* Return gamma (X), for positive X less than 184, in the form R *
   2^(*EXP2_ADJ), where R is the return value and *EXP2_ADJ is set to
   avoid overflow or underflow in intermediate calculations.  */

static double
gamma_positive (double x, int *exp2_adj)
{
  int local_signgam;
  if (x < 0.5)
    {
      *exp2_adj = 0;
      return __ieee754_exp (__ieee754_lgamma_r (x + 1, &local_signgam)) / x;
    }
  else if (x <= 1.5)
    {
      *exp2_adj = 0;
      return __ieee754_exp (__ieee754_lgamma_r (x, &local_signgam));
    }
  else if (x < 6.5)
    {
      /* Adjust into the range for using exp (lgamma).  */
      *exp2_adj = 0;
      double n = ceil (x - 1.5);
      double x_adj = x - n;
      double eps;
      double prod = __gamma_product (x_adj, 0, n, &eps);
      return (__ieee754_exp (__ieee754_lgamma_r (x_adj, &local_signgam))
	      * prod * (1.0 + eps));
    }
  else
    {
      double eps = 0;
      double x_eps = 0;
      double x_adj = x;
      double prod = 1;
      if (x < 12.0)
	{
	  /* Adjust into the range for applying Stirling's
	     approximation.  */
	  double n = ceil (12.0 - x);
	  x_adj = math_narrow_eval (x + n);
	  x_eps = (x - (x_adj - n));
	  prod = __gamma_product (x_adj - n, x_eps, n, &eps);
	}
      /* The result is now gamma (X_ADJ + X_EPS) / (PROD * (1 + EPS)).
	 Compute gamma (X_ADJ + X_EPS) using Stirling's approximation,
	 starting by computing pow (X_ADJ, X_ADJ) with a power of 2
	 factored out.  */
      double x_adj_int = round (x_adj);
      double x_adj_frac = x_adj - x_adj_int;
      int x_adj_log2;
      double x_adj_mant = __frexp (x_adj, &x_adj_log2);
      if (x_adj_mant < M_SQRT1_2)
	{
	  x_adj_log2--;
	  x_adj_mant *= 2.0;
	}
      *exp2_adj = x_adj_log2 * (int) x_adj_int;
      double h1, l1, h2, l2;
      mul_split (&h1, &l1, __ieee754_pow (x_adj_mant, x_adj),
			   __ieee754_exp2 (x_adj_log2 * x_adj_frac));
      mul_split (&h2, &l2, __ieee754_exp (-x_adj), sqrt (2 * M_PI / x_adj));
      mul_expansion (&h1, &l1, h1, l1, h2, l2);
      /* Divide by prod * (1 + eps).  */
      div_expansion (&h1, &l1, h1, l1, prod, prod * eps);
      double exp_adj = x_eps * __ieee754_log (x_adj);
      double bsum = gamma_coeff[NCOEFF - 1];
      double x_adj2 = x_adj * x_adj;
      for (size_t i = 1; i <= NCOEFF - 1; i++)
	bsum = bsum / x_adj2 + gamma_coeff[NCOEFF - 1 - i];
      exp_adj += bsum / x_adj;
      /* Now return (h1+l1) * exp(exp_adj), where exp_adj is small.  */
      l1 += h1 * __expm1 (exp_adj);
      return h1 + l1;
    }
}

double
__ieee754_gamma_r (double x, int *signgamp)
{
  int32_t hx;
  uint32_t lx;
  double ret;

  EXTRACT_WORDS (hx, lx, x);

  if (__glibc_unlikely (((hx & 0x7fffffff) | lx) == 0))
    {
      /* Return value for x == 0 is Inf with divide by zero exception.  */
      *signgamp = 0;
      return 1.0 / x;
    }
  if (__builtin_expect (hx < 0, 0)
      && (uint32_t) hx < 0xfff00000 && rint (x) == x)
    {
      /* Return value for integer x < 0 is NaN with invalid exception.  */
      *signgamp = 0;
      return (x - x) / (x - x);
    }
  if (__glibc_unlikely ((unsigned int) hx == 0xfff00000 && lx == 0))
    {
      /* x == -Inf.  According to ISO this is NaN.  */
      *signgamp = 0;
      return x - x;
    }
  if (__glibc_unlikely ((hx & 0x7ff00000) == 0x7ff00000))
    {
      /* Positive infinity (return positive infinity) or NaN (return
	 NaN).  */
      *signgamp = 0;
      return x + x;
    }

  if (x >= 172.0)
    {
      /* Overflow.  */
      *signgamp = 0;
      ret = math_narrow_eval (DBL_MAX * DBL_MAX);
      return ret;
    }
  else
    {
      SET_RESTORE_ROUND (FE_TONEAREST);
      if (x > 0.0)
	{
	  *signgamp = 0;
	  int exp2_adj;
	  double tret = gamma_positive (x, &exp2_adj);
	  ret = __scalbn (tret, exp2_adj);
	}
      else if (x >= -DBL_EPSILON / 4.0)
	{
	  *signgamp = 0;
	  ret = 1.0 / x;
	}
      else
	{
	  double tx = trunc (x);
	  *signgamp = (tx == 2.0 * trunc (tx / 2.0)) ? -1 : 1;
	  if (x <= -184.0)
	    /* Underflow.  */
	    ret = DBL_MIN * DBL_MIN;
	  else
	    {
	      double frac = tx - x;
	      if (frac > 0.5)
		frac = 1.0 - frac;
	      double sinpix = (frac <= 0.25
			       ? __sin (M_PI * frac)
			       : __cos (M_PI * (0.5 - frac)));
	      int exp2_adj;
	      double h1, l1, h2, l2;
	      h2 = gamma_positive (-x, &exp2_adj);
	      mul_split (&h1, &l1, sinpix, h2);
	      /* sinpix*gamma_positive(.) = h1 + l1 */
	      mul_split (&h2, &l2, h1, x);
	      /* h1*x = h2 + l2 */
	      /* (h1 + l1) * x = h1*x + l1*x = h2 + l2 + l1*x */
	      l2 += l1 * x;
	      /* x*sinpix*gamma_positive(.) ~ h2 + l2 */
	      h1 = 0x3.243f6a8885a3p+0;   /* binary64 approximation of Pi */
	      l1 = 0x8.d313198a2e038p-56; /* |h1+l1-Pi| < 3e-33 */
	      /* Now we divide h1 + l1 by h2 + l2.  */
	      div_expansion (&h1, &l1, h1, l1, h2, l2);
	      ret = __scalbn (-h1, -exp2_adj);
	      math_check_force_underflow_nonneg (ret);
	    }
	}
      ret = math_narrow_eval (ret);
    }
  if (isinf (ret) && x != 0)
    {
      if (*signgamp < 0)
	{
	  ret = math_narrow_eval (-copysign (DBL_MAX, ret) * DBL_MAX);
	  ret = -ret;
	}
      else
	ret = math_narrow_eval (copysign (DBL_MAX, ret) * DBL_MAX);
      return ret;
    }
  else if (ret == 0)
    {
      if (*signgamp < 0)
	{
	  ret = math_narrow_eval (-copysign (DBL_MIN, ret) * DBL_MIN);
	  ret = -ret;
	}
      else
	ret = math_narrow_eval (copysign (DBL_MIN, ret) * DBL_MIN);
      return ret;
    }
  else
    return ret;
}
libm_alias_finite (__ieee754_gamma_r, __gamma_r)
