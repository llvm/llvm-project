/* Compute a product of 1 + (T/X), 1 + (T/(X+1)), ....
   Copyright (C) 2015-2021 Free Software Foundation, Inc.
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

/* Compute the product of 1 + (T / (X + X_EPS)), 1 + (T / (X + X_EPS +
   1)), ..., 1 + (T / (X + X_EPS + N - 1)), minus 1.  X is such that
   all the values X + 1, ..., X + N - 1 are exactly representable, and
   X_EPS / X is small enough that factors quadratic in it can be
   neglected.  */

long double
__lgamma_productl (long double t, long double x, long double x_eps, int n)
{
  long double x_full = x + x_eps;
  long double ret = 0;
  for (int i = 0; i < n; i++)
    /* FIXME: no extra precision used.  */
    ret += (t / (x_full + i)) * (1 + ret);
  return ret;
}
