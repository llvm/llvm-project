/* Return arc hyperbolic tangent for a complex float type.
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
M_DECL_FUNC (__catanh) (CFLOAT x)
{
  CFLOAT res;
  int rcls = fpclassify (__real__ x);
  int icls = fpclassify (__imag__ x);

  if (__glibc_unlikely (rcls <= FP_INFINITE || icls <= FP_INFINITE))
    {
      if (icls == FP_INFINITE)
	{
	  __real__ res = M_COPYSIGN (0, __real__ x);
	  __imag__ res = M_COPYSIGN (M_MLIT (M_PI_2), __imag__ x);
	}
      else if (rcls == FP_INFINITE || rcls == FP_ZERO)
	{
	  __real__ res = M_COPYSIGN (0, __real__ x);
	  if (icls >= FP_ZERO)
	    __imag__ res = M_COPYSIGN (M_MLIT (M_PI_2), __imag__ x);
	  else
	    __imag__ res = M_NAN;
	}
      else
	{
	  __real__ res = M_NAN;
	  __imag__ res = M_NAN;
	}
    }
  else if (__glibc_unlikely (rcls == FP_ZERO && icls == FP_ZERO))
    {
      res = x;
    }
  else
    {
      if (M_FABS (__real__ x) >= 16 / M_EPSILON
	  || M_FABS (__imag__ x) >= 16 / M_EPSILON)
	{
	  __imag__ res = M_COPYSIGN (M_MLIT (M_PI_2), __imag__ x);
	  if (M_FABS (__imag__ x) <= 1)
	    __real__ res = 1 / __real__ x;
	  else if (M_FABS (__real__ x) <= 1)
	    __real__ res = __real__ x / __imag__ x / __imag__ x;
	  else
	    {
	      FLOAT h = M_HYPOT (__real__ x / 2, __imag__ x / 2);
	      __real__ res = __real__ x / h / h / 4;
	    }
	}
      else
	{
	  if (M_FABS (__real__ x) == 1
	      && M_FABS (__imag__ x) < M_EPSILON * M_EPSILON)
	    __real__ res = (M_COPYSIGN (M_LIT (0.5), __real__ x)
			    * ((FLOAT) M_MLIT (M_LN2)
			       - M_LOG (M_FABS (__imag__ x))));
	  else
	    {
	      FLOAT i2 = 0;
	      if (M_FABS (__imag__ x) >= M_EPSILON * M_EPSILON)
		i2 = __imag__ x * __imag__ x;

	      FLOAT num = 1 + __real__ x;
	      num = i2 + num * num;

	      FLOAT den = 1 - __real__ x;
	      den = i2 + den * den;

	      FLOAT f = num / den;
	      if (f < M_LIT (0.5))
		__real__ res = M_LIT (0.25) * M_LOG (f);
	      else
		{
		  num = 4 * __real__ x;
		  __real__ res = M_LIT (0.25) * M_LOG1P (num / den);
		}
	    }

	  FLOAT absx, absy, den;

	  absx = M_FABS (__real__ x);
	  absy = M_FABS (__imag__ x);
	  if (absx < absy)
	    {
	      FLOAT t = absx;
	      absx = absy;
	      absy = t;
	    }

	  if (absy < M_EPSILON / 2)
	    {
	      den = (1 - absx) * (1 + absx);
	      if (den == 0)
		den = 0;
	    }
	  else if (absx >= 1)
	    den = (1 - absx) * (1 + absx) - absy * absy;
	  else if (absx >= M_LIT (0.75) || absy >= M_LIT (0.5))
	    den = -M_SUF (__x2y2m1) (absx, absy);
	  else
	    den = (1 - absx) * (1 + absx) - absy * absy;

	  __imag__ res = M_LIT (0.5) * M_ATAN2 (2 * __imag__ x, den);
	}

      math_check_force_underflow_complex (res);
    }

  return res;
}

declare_mgen_alias (__catanh, catanh)
