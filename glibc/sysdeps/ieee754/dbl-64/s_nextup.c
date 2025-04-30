/* Return the least floating-point number greater than X.
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

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <libm-alias-double.h>

/* Return the least floating-point number greater than X.  */
double
__nextup (double x)
{
  int32_t hx, ix;
  uint32_t lx;

  EXTRACT_WORDS (hx, lx, x);
  ix = hx & 0x7fffffff;

  if (((ix >= 0x7ff00000) && ((ix - 0x7ff00000) | lx) != 0))  /* x is nan.  */
    return x + x;
  if ((ix | lx) == 0)
    return DBL_TRUE_MIN;
  if (hx >= 0)
    {				/* x > 0.  */
      if (isinf (x))
        return x;
      lx += 1;
      if (lx == 0)
        hx += 1;
    }
  else
    {				/* x < 0.  */
      if (lx == 0)
        hx -= 1;
      lx -= 1;
    }
  INSERT_WORDS (x, hx, lx);
  return x;
}

libm_alias_double (__nextup, nextup)
