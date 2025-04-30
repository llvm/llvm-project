/* Return arc hyperbolic sine for a complex float type.
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
M_DECL_FUNC (__casinh) (CFLOAT x)
{
  CFLOAT res;
  int rcls = fpclassify (__real__ x);
  int icls = fpclassify (__imag__ x);

  if (rcls <= FP_INFINITE || icls <= FP_INFINITE)
    {
      if (icls == FP_INFINITE)
	{
	  __real__ res = M_COPYSIGN (M_HUGE_VAL, __real__ x);

	  if (rcls == FP_NAN)
	    __imag__ res = M_NAN;
	  else
	    __imag__ res = M_COPYSIGN ((rcls >= FP_ZERO
				        ? M_MLIT (M_PI_2) : M_MLIT (M_PI_4)),
				       __imag__ x);
	}
      else if (rcls <= FP_INFINITE)
	{
	  __real__ res = __real__ x;
	  if ((rcls == FP_INFINITE && icls >= FP_ZERO)
	      || (rcls == FP_NAN && icls == FP_ZERO))
	    __imag__ res = M_COPYSIGN (0, __imag__ x);
	  else
	    __imag__ res = M_NAN;
	}
      else
	{
	  __real__ res = M_NAN;
	  __imag__ res = M_NAN;
	}
    }
  else if (rcls == FP_ZERO && icls == FP_ZERO)
    {
      res = x;
    }
  else
    {
      res = M_SUF (__kernel_casinh) (x, 0);
    }

  return res;
}

declare_mgen_alias (__casinh, casinh)
