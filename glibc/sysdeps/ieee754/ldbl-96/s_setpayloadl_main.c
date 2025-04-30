/* Set NaN payload.  ldbl-96 version.
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
#define PAYLOAD_DIG 62
#define EXPLICIT_MANT_DIG 63

int
FUNC (long double *x, long double payload)
{
  uint32_t hx, lx;
  uint16_t exponent;
  GET_LDOUBLE_WORDS (exponent, hx, lx, payload);
  /* Test if argument is (a) negative or too large; (b) too small,
     except for 0 when allowed; (c) not an integer.  */
  if (exponent >= BIAS + PAYLOAD_DIG
      || (exponent < BIAS && !(SET_HIGH_BIT
			       && exponent == 0 && hx == 0 && lx == 0)))
    {
      SET_LDOUBLE_WORDS (*x, 0, 0, 0);
      return 1;
    }
  int shift = BIAS + EXPLICIT_MANT_DIG - exponent;
  if (shift < 32
      ? (lx & ((1U << shift) - 1)) != 0
      : (lx != 0 || (hx & ((1U << (shift - 32)) - 1)) != 0))
    {
      SET_LDOUBLE_WORDS (*x, 0, 0, 0);
      return 1;
    }
  if (exponent != 0)
    {
      if (shift >= 32)
	{
	  lx = hx >> (shift - 32);
	  hx = 0;
	}
      else if (shift != 0)
	{
	  lx = (lx >> shift) | (hx << (32 - shift));
	  hx >>= shift;
	}
    }
  hx |= 0x80000000 | (SET_HIGH_BIT ? 0x40000000 : 0);
  SET_LDOUBLE_WORDS (*x, 0x7fff, hx, lx);
  return 0;
}
