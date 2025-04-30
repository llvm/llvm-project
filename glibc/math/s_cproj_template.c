/* Compute projection of complex float type value to Riemann sphere.
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
M_DECL_FUNC (__cproj) (CFLOAT x)
{
  if (isinf (__real__ x) || isinf (__imag__ x))
    {
      CFLOAT res;

      __real__ res = INFINITY;
      __imag__ res = M_COPYSIGN (0, __imag__ x);

      return res;
    }

  return x;
}

declare_mgen_alias (__cproj, cproj)
