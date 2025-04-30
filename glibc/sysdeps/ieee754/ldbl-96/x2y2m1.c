/* Compute x^2 + y^2 - 1, without large cancellation error.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library; if not, see
   <https://www.gnu.org/licenses/>.  */

#include <math.h>
#include <math_private.h>
#include <float.h>

#if FLT_EVAL_METHOD == 0

# include <sysdeps/ieee754/dbl-64/x2y2m1.c>

#else

/* Return X^2 + Y^2 - 1, computed without large cancellation error.
   It is given that 1 > X >= Y >= epsilon / 2, and that X^2 + Y^2 >=
   0.5.  */

double
__x2y2m1 (double x, double y)
{
  return (double) __x2y2m1l (x, y);
}

#endif
