/* Complex square root of a float type.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Based on an algorithm by Stephen L. Moshier <moshier@world.std.com>.
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
M_DECL_FUNC (__csqrt) (CFLOAT x)
{
  CFLOAT res;
  int rcls = fpclassify (__real__ x);
  int icls = fpclassify (__imag__ x);

  if (__glibc_unlikely (rcls <= FP_INFINITE || icls <= FP_INFINITE))
    {
      if (icls == FP_INFINITE)
	{
	  __real__ res = M_HUGE_VAL;
	  __imag__ res = __imag__ x;
	}
      else if (rcls == FP_INFINITE)
	{
	  if (__real__ x < 0)
	    {
	      __real__ res = icls == FP_NAN ? M_NAN : 0;
	      __imag__ res = M_COPYSIGN (M_HUGE_VAL, __imag__ x);
	    }
	  else
	    {
	      __real__ res = __real__ x;
	      __imag__ res = (icls == FP_NAN
			      ? M_NAN : M_COPYSIGN (0, __imag__ x));
	    }
	}
      else
	{
	  __real__ res = M_NAN;
	  __imag__ res = M_NAN;
	}
    }
  else
    {
      if (__glibc_unlikely (icls == FP_ZERO))
	{
	  if (__real__ x < 0)
	    {
	      __real__ res = 0;
	      __imag__ res = M_COPYSIGN (M_SQRT (-__real__ x), __imag__ x);
	    }
	  else
	    {
	      __real__ res = M_FABS (M_SQRT (__real__ x));
	      __imag__ res = M_COPYSIGN (0, __imag__ x);
	    }
	}
      else if (__glibc_unlikely (rcls == FP_ZERO))
	{
	  FLOAT r;
	  if (M_FABS (__imag__ x) >= 2 * M_MIN)
	    r = M_SQRT (M_LIT (0.5) * M_FABS (__imag__ x));
	  else
	    r = M_LIT (0.5) * M_SQRT (2 * M_FABS (__imag__ x));

	  __real__ res = r;
	  __imag__ res = M_COPYSIGN (r, __imag__ x);
	}
      else
	{
	  FLOAT d, r, s;
	  int scale = 0;

	  if (M_FABS (__real__ x) > M_MAX / 4)
	    {
	      scale = 1;
	      __real__ x = M_SCALBN (__real__ x, -2 * scale);
	      __imag__ x = M_SCALBN (__imag__ x, -2 * scale);
	    }
	  else if (M_FABS (__imag__ x) > M_MAX / 4)
	    {
	      scale = 1;
	      if (M_FABS (__real__ x) >= 4 * M_MIN)
		__real__ x = M_SCALBN (__real__ x, -2 * scale);
	      else
		__real__ x = 0;
	      __imag__ x = M_SCALBN (__imag__ x, -2 * scale);
	    }
	  else if (M_FABS (__real__ x) < 2 * M_MIN
		   && M_FABS (__imag__ x) < 2 * M_MIN)
	    {
	      scale = -((M_MANT_DIG + 1) / 2);
	      __real__ x = M_SCALBN (__real__ x, -2 * scale);
	      __imag__ x = M_SCALBN (__imag__ x, -2 * scale);
	    }

	  d = M_HYPOT (__real__ x, __imag__ x);
	  /* Use the identity   2  Re res  Im res = Im x
	     to avoid cancellation error in  d +/- Re x.  */
	  if (__real__ x > 0)
	    {
	      r = M_SQRT (M_LIT (0.5) * (d + __real__ x));
	      if (scale == 1 && M_FABS (__imag__ x) < 1)
		{
		  /* Avoid possible intermediate underflow.  */
		  s = __imag__ x / r;
		  r = M_SCALBN (r, scale);
		  scale = 0;
		}
	      else
		s = M_LIT (0.5) * (__imag__ x / r);
	    }
	  else
	    {
	      s = M_SQRT (M_LIT (0.5) * (d - __real__ x));
	      if (scale == 1 && M_FABS (__imag__ x) < 1)
		{
		  /* Avoid possible intermediate underflow.  */
		  r = M_FABS (__imag__ x / s);
		  s = M_SCALBN (s, scale);
		  scale = 0;
		}
	      else
		r = M_FABS (M_LIT (0.5) * (__imag__ x / s));
	    }

	  if (scale)
	    {
	      r = M_SCALBN (r, scale);
	      s = M_SCALBN (s, scale);
	    }

	  math_check_force_underflow (r);
	  math_check_force_underflow (s);

	  __real__ res = r;
	  __imag__ res = M_COPYSIGN (s, __imag__ x);
	}
    }

  return res;
}
declare_mgen_alias (__csqrt, csqrt)
