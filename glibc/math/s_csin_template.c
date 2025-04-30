/* Complex sine function for float types.
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
#include <fenv.h>
#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <float.h>

CFLOAT
M_DECL_FUNC (__csin) (CFLOAT x)
{
  CFLOAT retval;
  int negate = signbit (__real__ x);
  int rcls = fpclassify (__real__ x);
  int icls = fpclassify (__imag__ x);

  __real__ x = M_FABS (__real__ x);

  if (__glibc_likely (icls >= FP_ZERO))
    {
      /* Imaginary part is finite.  */
      if (__glibc_likely (rcls >= FP_ZERO))
	{
	  /* Real part is finite.  */
	  const int t = (int) ((M_MAX_EXP - 1) * M_MLIT (M_LN2));
	  FLOAT sinix, cosix;

	  if (__glibc_likely (__real__ x > M_MIN))
	    {
	      M_SINCOS (__real__ x, &sinix, &cosix);
	    }
	  else
	    {
	      sinix = __real__ x;
	      cosix = 1;
	    }

	  if (negate)
	    sinix = -sinix;

	  if (M_FABS (__imag__ x) > t)
	    {
	      FLOAT exp_t = M_EXP (t);
	      FLOAT ix = M_FABS (__imag__ x);
	      if (signbit (__imag__ x))
		cosix = -cosix;
	      ix -= t;
	      sinix *= exp_t / 2;
	      cosix *= exp_t / 2;
	      if (ix > t)
		{
		  ix -= t;
		  sinix *= exp_t;
		  cosix *= exp_t;
		}
	      if (ix > t)
		{
		  /* Overflow (original imaginary part of x > 3t).  */
		  __real__ retval = M_MAX * sinix;
		  __imag__ retval = M_MAX * cosix;
		}
	      else
		{
		  FLOAT exp_val = M_EXP (ix);
		  __real__ retval = exp_val * sinix;
		  __imag__ retval = exp_val * cosix;
		}
	    }
	  else
	    {
	      __real__ retval = M_COSH (__imag__ x) * sinix;
	      __imag__ retval = M_SINH (__imag__ x) * cosix;
	    }

	  math_check_force_underflow_complex (retval);
	}
      else
	{
	  if (icls == FP_ZERO)
	    {
	      /* Imaginary part is 0.0.  */
	      __real__ retval = __real__ x - __real__ x;
	      __imag__ retval = __imag__ x;
	    }
	  else
	    {
	      __real__ retval = M_NAN;
	      __imag__ retval = M_NAN;

	      feraiseexcept (FE_INVALID);
	    }
	}
    }
  else if (icls == FP_INFINITE)
    {
      /* Imaginary part is infinite.  */
      if (rcls == FP_ZERO)
	{
	  /* Real part is 0.0.  */
	  __real__ retval = M_COPYSIGN (0, negate ? -1 : 1);
	  __imag__ retval = __imag__ x;
	}
      else if (rcls > FP_ZERO)
	{
	  /* Real part is finite.  */
	  FLOAT sinix, cosix;

	  if (__glibc_likely (__real__ x > M_MIN))
	    {
	      M_SINCOS (__real__ x, &sinix, &cosix);
	    }
	  else
	    {
	      sinix = __real__ x;
	      cosix = 1;
	    }

	  __real__ retval = M_COPYSIGN (M_HUGE_VAL, sinix);
	  __imag__ retval = M_COPYSIGN (M_HUGE_VAL, cosix);

	  if (negate)
	    __real__ retval = -__real__ retval;
	  if (signbit (__imag__ x))
	    __imag__ retval = -__imag__ retval;
	}
      else
	{
	  __real__ retval = __real__ x - __real__ x;
	  __imag__ retval = M_HUGE_VAL;
	}
    }
  else
    {
      if (rcls == FP_ZERO)
	__real__ retval = M_COPYSIGN (0, negate ? -1 : 1);
      else
	__real__ retval = M_NAN;
      __imag__ retval = M_NAN;
    }

  return retval;
}

declare_mgen_alias (__csin, csin)
