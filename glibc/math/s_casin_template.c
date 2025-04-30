/* Return arc sine of a complex float type.
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
M_DECL_FUNC (__casin) (CFLOAT x)
{
  CFLOAT res;

  if (isnan (__real__ x) || isnan (__imag__ x))
    {
      if (__real__ x == 0)
	{
	  res = x;
	}
      else if (isinf (__real__ x) || isinf (__imag__ x))
	{
	  __real__ res = M_NAN;
	  __imag__ res = M_COPYSIGN (M_HUGE_VAL, __imag__ x);
	}
      else
	{
	  __real__ res = M_NAN;
	  __imag__ res = M_NAN;
	}
    }
  else
    {
      CFLOAT y;

      __real__ y = -__imag__ x;
      __imag__ y = __real__ x;

      y = M_SUF (__casinh) (y);

      __real__ res = __imag__ y;
      __imag__ res = -__real__ y;
    }

  return res;
}

declare_mgen_alias (__casin, casin)
