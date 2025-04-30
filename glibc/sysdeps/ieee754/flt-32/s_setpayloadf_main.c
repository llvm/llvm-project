/* Set NaN payload.  flt-32 version.
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
#include <libm-alias-float.h>
#include <nan-high-order-bit.h>
#include <stdint.h>

#define SET_HIGH_BIT (HIGH_ORDER_BIT_IS_SET_FOR_SNAN ? SIG : !SIG)
#define BIAS 0x7f
#define PAYLOAD_DIG 22
#define EXPLICIT_MANT_DIG 23

int
FUNC (float *x, float payload)
{
  uint32_t ix;
  GET_FLOAT_WORD (ix, payload);
  int exponent = ix >> EXPLICIT_MANT_DIG;
  /* Test if argument is (a) negative or too large; (b) too small,
     except for 0 when allowed; (c) not an integer.  */
  if (exponent >= BIAS + PAYLOAD_DIG
      || (exponent < BIAS && !(SET_HIGH_BIT && ix == 0))
      || (ix & ((1U << (BIAS + EXPLICIT_MANT_DIG - exponent)) - 1)) != 0)
    {
      SET_FLOAT_WORD (*x, 0);
      return 1;
    }
  if (ix != 0)
    {
      ix &= (1U << EXPLICIT_MANT_DIG) - 1;
      ix |= 1U << EXPLICIT_MANT_DIG;
      ix >>= BIAS + EXPLICIT_MANT_DIG - exponent;
    }
  ix |= 0x7f800000 | (SET_HIGH_BIT ? 0x400000 : 0);
  SET_FLOAT_WORD (*x, ix);
  return 0;
}
