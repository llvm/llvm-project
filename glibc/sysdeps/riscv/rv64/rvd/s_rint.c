/* rint().  RISC-V version.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#define NO_MATH_REDIRECT
#include <math.h>
#include <stdbool.h>
#include <libm-alias-double.h>
#include <stdint.h>

double
__rint (double x)
{
  bool nan;
  double mag;

  nan = isnan (x);
  mag = fabs (x);

  if (nan)
    return x + x;

  if (mag < (1ULL << __DBL_MANT_DIG__))
    {
      int64_t i;
      double new_x;

      asm ("fcvt.l.d %0, %1" : "=r" (i) : "f" (x));
      asm ("fcvt.d.l %0, %1" : "=f" (new_x) : "r" (i));

      /* rint(-0) == -0, and in general we'll always have the same
	 sign as our input.  */
      x = copysign (new_x, x);
    }

  return x;
}

libm_alias_double (__rint, rint)
