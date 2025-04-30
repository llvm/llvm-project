/* Return arc hyperbolic cosine for a complex type.
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


CFLOAT
M_DECL_FUNC (__cacosh) (CFLOAT x)
{
  CFLOAT res;
  int rcls = fpclassify (__real__ x);
  int icls = fpclassify (__imag__ x);

  if (rcls <= FP_INFINITE || icls <= FP_INFINITE)
    {
      if (icls == FP_INFINITE)
	{
	  __real__ res = M_HUGE_VAL;

	  if (rcls == FP_NAN)
	    __imag__ res = M_NAN;
	  else
	    __imag__ res = M_COPYSIGN ((rcls == FP_INFINITE
					? (__real__ x < 0
					   ? M_MLIT (M_PI) - M_MLIT (M_PI_4)
					   : M_MLIT (M_PI_4))
					: M_MLIT (M_PI_2)), __imag__ x);
	}
      else if (rcls == FP_INFINITE)
	{
	  __real__ res = M_HUGE_VAL;

	  if (icls >= FP_ZERO)
	    __imag__ res = M_COPYSIGN (signbit (__real__ x)
				       ? M_MLIT (M_PI) : 0, __imag__ x);
	  else
	    __imag__ res = M_NAN;
	}
      else
	{
	  __real__ res = M_NAN;
	  if (rcls == FP_ZERO)
	    __imag__ res = M_MLIT (M_PI_2);
	  else
	    __imag__ res = M_NAN;
	}
    }
  else if (rcls == FP_ZERO && icls == FP_ZERO)
    {
      __real__ res = 0;
      __imag__ res = M_COPYSIGN (M_MLIT (M_PI_2), __imag__ x);
    }
  else
    {
      CFLOAT y;

      __real__ y = -__imag__ x;
      __imag__ y = __real__ x;

      y = M_SUF (__kernel_casinh) (y, 1);

      if (signbit (__imag__ x))
	{
	  __real__ res = __real__ y;
	  __imag__ res = -__imag__ y;
	}
      else
	{
	  __real__ res = -__real__ y;
	  __imag__ res = __imag__ y;
	}
    }

  return res;
}

declare_mgen_alias (__cacosh, cacosh)
