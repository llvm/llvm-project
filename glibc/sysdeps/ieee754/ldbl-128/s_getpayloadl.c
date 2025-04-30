/* Get NaN payload.  ldbl-128 version.
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
#include <math_private.h>
#include <libm-alias-ldouble.h>
#include <stdint.h>

_Float128
__getpayloadl (const _Float128 *x)
{
  uint64_t hx, lx;
  GET_LDOUBLE_WORDS64 (hx, lx, *x);
  if ((hx & 0x7fff000000000000ULL) != 0x7fff000000000000ULL
      || ((hx & 0xffffffffffffULL) | lx) == 0)
    return -1;
  hx &= 0x7fffffffffffULL;
  /* Construct the representation of the return value directly, since
     128-bit integers may not be available.  */
  int lz;
  if (hx == 0)
    {
      if (lx == 0)
	return 0.0L;
      else
	lz = __builtin_clzll (lx) + 64;
    }
  else
    lz = __builtin_clzll (hx);
  int shift = lz - 15;
  if (shift >= 64)
    {
      hx = lx << (shift - 64);
      lx = 0;
    }
  else
    {
      /* 2 <= SHIFT <= 63.  */
      hx = (hx << shift) | (lx >> (64 - shift));
      lx <<= shift;
    }
  hx = (hx & 0xffffffffffffULL) | ((0x3fffULL + 127 - lz) << 48);
  _Float128 ret;
  SET_LDOUBLE_WORDS64 (ret, hx, lx);
  return ret;
}
libm_alias_ldouble (__getpayload, getpayload)
