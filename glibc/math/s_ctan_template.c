/* Complex tangent function for a complex float type.
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
M_DECL_FUNC (__ctan) (CFLOAT x)
{
  CFLOAT res;

  if (__glibc_unlikely (!isfinite (__real__ x) || !isfinite (__imag__ x)))
    {
      if (isinf (__imag__ x))
	{
	  if (isfinite (__real__ x) && M_FABS (__real__ x) > 1)
	    {
	      FLOAT sinrx, cosrx;
	      M_SINCOS (__real__ x, &sinrx, &cosrx);
	      __real__ res = M_COPYSIGN (0, sinrx * cosrx);
	    }
	  else
	    __real__ res = M_COPYSIGN (0, __real__ x);
	  __imag__ res = M_COPYSIGN (1, __imag__ x);
	}
      else if (__real__ x == 0)
	{
	  res = x;
	}
      else
	{
	  __real__ res = M_NAN;
	  if (__imag__ x == 0)
	    __imag__ res = __imag__ x;
	  else
	    __imag__ res = M_NAN;

	  if (isinf (__real__ x))
	    feraiseexcept (FE_INVALID);
	}
    }
  else
    {
      FLOAT sinrx, cosrx;
      FLOAT den;
      const int t = (int) ((M_MAX_EXP - 1) * M_MLIT (M_LN2) / 2);

      /* tan(x+iy) = (sin(2x) + i*sinh(2y))/(cos(2x) + cosh(2y))
	 = (sin(x)*cos(x) + i*sinh(y)*cosh(y)/(cos(x)^2 + sinh(y)^2). */

      if (__glibc_likely (M_FABS (__real__ x) > M_MIN))
	{
	  M_SINCOS (__real__ x, &sinrx, &cosrx);
	}
      else
	{
	  sinrx = __real__ x;
	  cosrx = 1;
	}

      if (M_FABS (__imag__ x) > t)
	{
	  /* Avoid intermediate overflow when the real part of the
	     result may be subnormal.  Ignoring negligible terms, the
	     imaginary part is +/- 1, the real part is
	     sin(x)*cos(x)/sinh(y)^2 = 4*sin(x)*cos(x)/exp(2y).  */
	  FLOAT exp_2t = M_EXP (2 * t);

	  __imag__ res = M_COPYSIGN (1, __imag__ x);
	  __real__ res = 4 * sinrx * cosrx;
	  __imag__ x = M_FABS (__imag__ x);
	  __imag__ x -= t;
	  __real__ res /= exp_2t;
	  if (__imag__ x > t)
	    {
	      /* Underflow (original imaginary part of x has absolute
		 value > 2t).  */
	      __real__ res /= exp_2t;
	    }
	  else
	    __real__ res /= M_EXP (2 * __imag__ x);
	}
      else
	{
	  FLOAT sinhix, coshix;
	  if (M_FABS (__imag__ x) > M_MIN)
	    {
	      sinhix = M_SINH (__imag__ x);
	      coshix = M_COSH (__imag__ x);
	    }
	  else
	    {
	      sinhix = __imag__ x;
	      coshix = 1;
	    }

	  if (M_FABS (sinhix) > M_FABS (cosrx) * M_EPSILON)
	    den = cosrx * cosrx + sinhix * sinhix;
	  else
	    den = cosrx * cosrx;
	  __real__ res = sinrx * cosrx / den;
	  __imag__ res = sinhix * coshix / den;
	}
      math_check_force_underflow_complex (res);
    }

  return res;
}

declare_mgen_alias (__ctan, ctan)
