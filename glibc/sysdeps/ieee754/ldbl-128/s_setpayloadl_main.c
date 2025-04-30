/* Set NaN payload.  ldbl-128 version.
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
#include <nan-high-order-bit.h>
#include <stdint.h>

#define SET_HIGH_BIT (HIGH_ORDER_BIT_IS_SET_FOR_SNAN ? SIG : !SIG)
#define BIAS 0x3fff
#define PAYLOAD_DIG 111
#define EXPLICIT_MANT_DIG 112

int
FUNC (_Float128 *x, _Float128 payload)
{
  uint64_t hx, lx;
  GET_LDOUBLE_WORDS64 (hx, lx, payload);
  int exponent = hx >> (EXPLICIT_MANT_DIG - 64);
  /* Test if argument is (a) negative or too large; (b) too small,
     except for 0 when allowed; (c) not an integer.  */
  if (exponent >= BIAS + PAYLOAD_DIG
      || (exponent < BIAS && !(SET_HIGH_BIT && hx == 0 && lx == 0)))
    {
      SET_LDOUBLE_WORDS64 (*x, 0, 0);
      return 1;
    }
  int shift = BIAS + EXPLICIT_MANT_DIG - exponent;
  if (shift < 64
      ? (lx & ((1ULL << shift) - 1)) != 0
      : (lx != 0 || (hx & ((1ULL << (shift - 64)) - 1)) != 0))
    {
      SET_LDOUBLE_WORDS64 (*x, 0, 0);
      return 1;
    }
  if (exponent != 0)
    {
      hx &= (1ULL << (EXPLICIT_MANT_DIG - 64)) - 1;
      hx |= 1ULL << (EXPLICIT_MANT_DIG - 64);
      if (shift >= 64)
	{
	  lx = hx >> (shift - 64);
	  hx = 0;
	}
      else if (shift != 0)
	{
	  lx = (lx >> shift) | (hx << (64 - shift));
	  hx >>= shift;
	}
    }
  hx |= 0x7fff000000000000ULL | (SET_HIGH_BIT ? 0x800000000000ULL : 0);
  SET_LDOUBLE_WORDS64 (*x, hx, lx);
  return 0;
}
