/* Return value with minimum magnitude.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

FLOAT
M_DECL_FUNC (__fminmag) (FLOAT x, FLOAT y)
{
  FLOAT ax = M_FABS (x);
  FLOAT ay = M_FABS (y);
  if (isless (ax, ay))
    return x;
  else if (isgreater (ax, ay))
    return y;
  else if (ax == ay)
    return x < y ? x : y;
  else if (issignaling (x) || issignaling (y))
    return x + y;
  else
    return isnan (y) ? x : y;
}

declare_mgen_alias (__fminmag, fminmag);
