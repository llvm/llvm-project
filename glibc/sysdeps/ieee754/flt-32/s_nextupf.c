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
#include <libm-alias-float.h>

/* Return the least floating-point number greater than X.  */
float
__nextupf (float x)
{
  int32_t hx, ix;

  GET_FLOAT_WORD (hx, x);
  ix = hx & 0x7fffffff;
  if (ix == 0)
    return FLT_TRUE_MIN;
  if (ix > 0x7f800000)		/* x is nan.  */
    return x + x;
  if (hx >= 0)
    {				/* x > 0.  */
      if (isinf (x))
        return x;
      hx += 1;
    }
  else
    hx -= 1;
  SET_FLOAT_WORD (x, hx);
  return x;
}

libm_alias_float (__nextup, nextup)
