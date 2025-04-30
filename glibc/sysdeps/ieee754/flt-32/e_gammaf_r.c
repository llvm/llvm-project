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

/* Coefficients B_2k / 2k(2k-1) of x^-(2k-1) inside exp in Stirling's
   approximation to gamma function.  */

static const float gamma_coeff[] =
  {
    0x1.555556p-4f,
    -0xb.60b61p-12f,
    0x3.403404p-12f,
  };

#define NCOEFF (sizeof (gamma_coeff) / sizeof (gamma_coeff[0]))

/* Return gamma (X), for positive X less than 42, in the form R *
   2^(*EXP2_ADJ), where R is the return value and *EXP2_ADJ is set to
   avoid overflow or underflow in intermediate calculations.  */

static float
gammaf_positive (float x, int *exp2_adj)
{
  int local_signgam;
  if (x < 0.5f)
    {
      *exp2_adj = 0;
      return __ieee754_expf (__ieee754_lgammaf_r (x + 1, &local_signgam)) / x;
    }
  else if (x <= 1.5f)
    {
      *exp2_adj = 0;
      return __ieee754_expf (__ieee754_lgammaf_r (x, &local_signgam));
    }
  else if (x < 2.5f)
    {
      *exp2_adj = 0;
      float x_adj = x - 1;
      return (__ieee754_expf (__ieee754_lgammaf_r (x_adj, &local_signgam))
	      * x_adj);
    }
  else
    {
      float eps = 0;
      float x_eps = 0;
      float x_adj = x;
      float prod = 1;
      if (x < 4.0f)
	{
	  /* Adjust into the range for applying Stirling's
	     approximation.  */
	  float n = ceilf (4.0f - x);
	  x_adj = math_narrow_eval (x + n);
	  x_eps = (x - (x_adj - n));
	  prod = __gamma_productf (x_adj - n, x_eps, n, &eps);
	}
      /* The result is now gamma (X_ADJ + X_EPS) / (PROD * (1 + EPS)).
	 Compute gamma (X_ADJ + X_EPS) using Stirling's approximation,
	 starting by computing pow (X_ADJ, X_ADJ) with a power of 2
	 factored out.  */
      float exp_adj = -eps;
      float x_adj_int = roundf (x_adj);
      float x_adj_frac = x_adj - x_adj_int;
      int x_adj_log2;
      float x_adj_mant = __frexpf (x_adj, &x_adj_log2);
      if (x_adj_mant < (float) M_SQRT1_2)
	{
	  x_adj_log2--;
	  x_adj_mant *= 2.0f;
	}
      *exp2_adj = x_adj_log2 * (int) x_adj_int;
      float ret = (__ieee754_powf (x_adj_mant, x_adj)
		   * __ieee754_exp2f (x_adj_log2 * x_adj_frac)
		   * __ieee754_expf (-x_adj)
		   * sqrtf (2 * (float) M_PI / x_adj)
		   / prod);
      exp_adj += x_eps * __ieee754_logf (x_adj);
      float bsum = gamma_coeff[NCOEFF - 1];
      float x_adj2 = x_adj * x_adj;
      for (size_t i = 1; i <= NCOEFF - 1; i++)
	bsum = bsum / x_adj2 + gamma_coeff[NCOEFF - 1 - i];
      exp_adj += bsum / x_adj;
      return ret + ret * __expm1f (exp_adj);
    }
}

float
__ieee754_gammaf_r (float x, int *signgamp)
{
  int32_t hx;
  float ret;

  GET_FLOAT_WORD (hx, x);

  if (__glibc_unlikely ((hx & 0x7fffffff) == 0))
    {
      /* Return value for x == 0 is Inf with divide by zero exception.  */
      *signgamp = 0;
      return 1.0 / x;
    }
  if (__builtin_expect (hx < 0, 0)
      && (uint32_t) hx < 0xff800000 && rintf (x) == x)
    {
      /* Return value for integer x < 0 is NaN with invalid exception.  */
      *signgamp = 0;
      return (x - x) / (x - x);
    }
  if (__glibc_unlikely (hx == 0xff800000))
    {
      /* x == -Inf.  According to ISO this is NaN.  */
      *signgamp = 0;
      return x - x;
    }
  if (__glibc_unlikely ((hx & 0x7f800000) == 0x7f800000))
    {
      /* Positive infinity (return positive infinity) or NaN (return
	 NaN).  */
      *signgamp = 0;
      return x + x;
    }

  if (x >= 36.0f)
    {
      /* Overflow.  */
      *signgamp = 0;
      ret = math_narrow_eval (FLT_MAX * FLT_MAX);
      return ret;
    }
  else
    {
      SET_RESTORE_ROUNDF (FE_TONEAREST);
      if (x > 0.0f)
	{
	  *signgamp = 0;
	  int exp2_adj;
	  float tret = gammaf_positive (x, &exp2_adj);
	  ret = __scalbnf (tret, exp2_adj);
	}
      else if (x >= -FLT_EPSILON / 4.0f)
	{
	  *signgamp = 0;
	  ret = 1.0f / x;
	}
      else
	{
	  float tx = truncf (x);
	  *signgamp = (tx == 2.0f * truncf (tx / 2.0f)) ? -1 : 1;
	  if (x <= -42.0f)
	    /* Underflow.  */
	    ret = FLT_MIN * FLT_MIN;
	  else
	    {
	      float frac = tx - x;
	      if (frac > 0.5f)
		frac = 1.0f - frac;
	      float sinpix = (frac <= 0.25f
			      ? __sinf ((float) M_PI * frac)
			      : __cosf ((float) M_PI * (0.5f - frac)));
	      int exp2_adj;
	      float tret = (float) M_PI / (-x * sinpix
					   * gammaf_positive (-x, &exp2_adj));
	      ret = __scalbnf (tret, -exp2_adj);
	      math_check_force_underflow_nonneg (ret);
	    }
	}
      ret = math_narrow_eval (ret);
    }
  if (isinf (ret) && x != 0)
    {
      if (*signgamp < 0)
	{
	  ret = math_narrow_eval (-copysignf (FLT_MAX, ret) * FLT_MAX);
	  ret = -ret;
	}
      else
	ret = math_narrow_eval (copysignf (FLT_MAX, ret) * FLT_MAX);
      return ret;
    }
  else if (ret == 0)
    {
      if (*signgamp < 0)
	{
	  ret = math_narrow_eval (-copysignf (FLT_MIN, ret) * FLT_MIN);
	  ret = -ret;
	}
      else
	ret = math_narrow_eval (copysignf (FLT_MIN, ret) * FLT_MIN);
      return ret;
    }
  else
    return ret;
}
libm_alias_finite (__ieee754_gammaf_r, __gammaf_r)
