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
#include <libm-alias-ldouble.h>

/* Return the least floating-point number greater than X.  */
_Float128
__nextupl (_Float128 x)
{
  int64_t hx, ix;
  uint64_t lx;

  GET_LDOUBLE_WORDS64 (hx, lx, x);
  ix = hx & 0x7fffffffffffffffLL;

  /* x is nan.  */
  if (((ix >= 0x7fff000000000000LL)
       && ((ix - 0x7fff000000000000LL) | lx) != 0))
    return x + x;
  if ((ix | lx) == 0)
    return LDBL_TRUE_MIN;
  if (hx >= 0)
    {				/* x > 0.  */
      if (isinf (x))
        return x;
      lx++;
      if (lx == 0)
        hx++;
    }
  else
    {				/* x < 0.  */
      if (lx == 0)
        hx--;
      lx--;
    }
  SET_LDOUBLE_WORDS64 (x, hx, lx);
  return x;
}

libm_alias_ldouble (__nextup, nextup)
