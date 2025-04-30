/* Return cosine of a complex type.
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
M_DECL_FUNC (__cacos) (CFLOAT x)
{
  CFLOAT y;
  CFLOAT res;
  int rcls = fpclassify (__real__ x);
  int icls = fpclassify (__imag__ x);

  if (rcls <= FP_INFINITE || icls <= FP_INFINITE
      || (rcls == FP_ZERO && icls == FP_ZERO))
    {
      y = M_SUF (__casin) (x);

      __real__ res = (FLOAT) M_MLIT (M_PI_2) - __real__ y;
      if (__real__ res == 0)
	__real__ res = 0;
      __imag__ res = -__imag__ y;
    }
  else
    {
      __real__ y = -__imag__ x;
      __imag__ y = __real__ x;

      y = M_SUF (__kernel_casinh) (y, 1);

      __real__ res = __imag__ y;
      __imag__ res = __real__ y;
    }

  return res;
}

declare_mgen_alias (__cacos, cacos);
