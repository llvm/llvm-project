/* Get NaN payload.  flt-32 version.
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

#include <fix-int-fp-convert-zero.h>
#include <math.h>
#include <math_private.h>
#include <libm-alias-float.h>
#include <stdint.h>

float
__getpayloadf (const float *x)
{
  uint32_t ix;
  GET_FLOAT_WORD (ix, *x);
  if ((ix & 0x7f800000) != 0x7f800000
      || (ix & 0x7fffff) == 0)
    return -1;
  ix &= 0x3fffff;
  if (FIX_INT_FP_CONVERT_ZERO && ix == 0)
    return 0.0f;
  return (float) ix;
}
libm_alias_float (__getpayload, getpayload)
