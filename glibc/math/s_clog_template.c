/* Compute complex natural logarithm.
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

#include <complex.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <float.h>

CFLOAT
M_DECL_FUNC (__clog) (CFLOAT x)
{
  CFLOAT result;
  int rcls = fpclassify (__real__ x);
  int icls = fpclassify (__imag__ x);

  if (__glibc_unlikely (rcls == FP_ZERO && icls == FP_ZERO))
    {
      /* Real and imaginary part are 0.0.  */
      __imag__ result = signbit (__real__ x) ? (FLOAT) M_MLIT (M_PI) : 0;
      __imag__ result = M_COPYSIGN (__imag__ result, __imag__ x);
      /* Yes, the following line raises an exception.  */
      __real__ result = -1 / M_FABS (__real__ x);
    }
  else if (__glibc_likely (rcls != FP_NAN && icls != FP_NAN))
    {
      /* Neither real nor imaginary part is NaN.  */
      FLOAT absx = M_FABS (__real__ x), absy = M_FABS (__imag__ x);
      int scale = 0;

      if (absx < absy)
	{
	  FLOAT t = absx;
	  absx = absy;
	  absy = t;
	}

      if (absx > M_MAX / 2)
	{
	  scale = -1;
	  absx = M_SCALBN (absx, scale);
	  absy = (absy >= M_MIN * 2 ? M_SCALBN (absy, scale) : 0);
	}
      else if (absx < M_MIN && absy < M_MIN)
	{
	  scale = M_MANT_DIG;
	  absx = M_SCALBN (absx, scale);
	  absy = M_SCALBN (absy, scale);
	}

      if (absx == 1 && scale == 0)
	{
	  __real__ result = M_LOG1P (absy * absy) / 2;
	  math_check_force_underflow_nonneg (__real__ result);
	}
      else if (absx > 1 && absx < 2 && absy < 1 && scale == 0)
	{
	  FLOAT d2m1 = (absx - 1) * (absx + 1);
	  if (absy >= M_EPSILON)
	    d2m1 += absy * absy;
	  __real__ result = M_LOG1P (d2m1) / 2;
	}
      else if (absx < 1
	       && absx >= M_LIT (0.5)
	       && absy < M_EPSILON / 2
	       && scale == 0)
	{
	  FLOAT d2m1 = (absx - 1) * (absx + 1);
	  __real__ result = M_LOG1P (d2m1) / 2;
	}
      else if (absx < 1
	       && absx >= M_LIT (0.5)
	       && scale == 0
	       && absx * absx + absy * absy >= M_LIT (0.5))
	{
	  FLOAT d2m1 = M_SUF (__x2y2m1) (absx, absy);
	  __real__ result = M_LOG1P (d2m1) / 2;
	}
      else
	{
	  FLOAT d = M_HYPOT (absx, absy);
	  __real__ result = M_LOG (d) - scale * (FLOAT) M_MLIT (M_LN2);
	}

      __imag__ result = M_ATAN2 (__imag__ x, __real__ x);
    }
  else
    {
      __imag__ result = M_NAN;
      if (rcls == FP_INFINITE || icls == FP_INFINITE)
	/* Real or imaginary part is infinite.  */
	__real__ result = M_HUGE_VAL;
      else
	__real__ result = M_NAN;
    }

  return result;
}

declare_mgen_alias (__clog, clog)
