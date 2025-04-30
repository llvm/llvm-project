/* Convert a number in multi-precision format to _Float128.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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

#include "gmp.h"
#include "gmp-impl.h"
#include <float.h>
#include <math.h>

#include "ieee754_float128.h"

/* Convert a multi-precision integer of the needed number of bits (113 for
   binary128) and an integral power of two to a `_Float128' in IEEE854
   quad-precision format.  */

_Float128
__mpn_construct_float128 (mp_srcptr frac_ptr, int expt, int sign)
{
  union ieee854_float128 u;

  u.ieee.negative = sign;
  u.ieee.exponent = expt + IEEE854_FLOAT128_BIAS;
#if BITS_PER_MP_LIMB == 32
  u.ieee.mantissa3 = frac_ptr[0];
  u.ieee.mantissa2 = frac_ptr[1];
  u.ieee.mantissa1 = frac_ptr[2];
  u.ieee.mantissa0 = frac_ptr[3] & (((mp_limb_t) 1
				     << (FLT128_MANT_DIG - 96)) - 1);
#elif BITS_PER_MP_LIMB == 64
  u.ieee.mantissa3 = frac_ptr[0] & (((mp_limb_t) 1 << 32) - 1);
  u.ieee.mantissa2 = frac_ptr[0] >> 32;
  u.ieee.mantissa1 = frac_ptr[1] & (((mp_limb_t) 1 << 32) - 1);
  u.ieee.mantissa0 = (frac_ptr[1] >> 32) & (((mp_limb_t) 1
					     << (FLT128_MANT_DIG - 96)) - 1);
#else
  #error "mp_limb size " BITS_PER_MP_LIMB "not accounted for"
#endif

  return u.d;
}
