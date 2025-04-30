/* Set NaN payload.  ldbl-128ibm version.
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
#include <nan-high-order-bit.h>
#include <stdint.h>

#define SET_HIGH_BIT (HIGH_ORDER_BIT_IS_SET_FOR_SNAN ? SIG : !SIG)
#define BIAS 0x3ff
#define PAYLOAD_DIG 51
#define EXPLICIT_MANT_DIG 52

int
FUNC (long double *x, long double payload)
{
  double hi, lo;
  uint64_t hx, lx;

  ldbl_unpack (payload, &hi, &lo);
  EXTRACT_WORDS64 (hx, hi);
  EXTRACT_WORDS64 (lx, lo);
  int exponent = hx >> EXPLICIT_MANT_DIG;
  /* Test if argument is (a) negative or too large; (b) too small,
     except for 0 when allowed; (c) not an integer.  All valid
     arguments have the low part zero.  */
  if ((lx & 0x7fffffffffffffffULL) != 0
      || exponent >= BIAS + PAYLOAD_DIG
      || (exponent < BIAS && !(SET_HIGH_BIT && hx == 0))
      || (hx & ((1ULL << (BIAS + EXPLICIT_MANT_DIG - exponent)) - 1)) != 0)
    {
      *x = 0.0L;
      return 1;
    }
  if (hx != 0)
    {
      hx &= (1ULL << EXPLICIT_MANT_DIG) - 1;
      hx |= 1ULL << EXPLICIT_MANT_DIG;
      hx >>= BIAS + EXPLICIT_MANT_DIG - exponent;
    }
  hx |= 0x7ff0000000000000ULL | (SET_HIGH_BIT ? 0x8000000000000ULL : 0);
  INSERT_WORDS64 (hi, hx);
  *x = ldbl_pack (hi, 0.0);
  return 0;
}
