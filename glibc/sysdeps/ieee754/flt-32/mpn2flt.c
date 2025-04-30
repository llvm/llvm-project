/* Copyright (C) 1995-2021 Free Software Foundation, Inc.
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
#include <ieee754.h>
#include <float.h>

/* Convert a multi-precision integer of the needed number of bits (24 for
   float) and an integral power of two to a `float' in IEEE754 single-
   precision format.  */

float
__mpn_construct_float (mp_srcptr frac_ptr, int expt, int sign)
{
  union ieee754_float u;

  u.ieee.negative = sign;
  u.ieee.exponent = expt + IEEE754_FLOAT_BIAS;
#if BITS_PER_MP_LIMB > FLT_MANT_DIG
  u.ieee.mantissa = frac_ptr[0] & (((mp_limb_t) 1 << FLT_MANT_DIG) - 1);
#else
  #error "mp_limb size " BITS_PER_MP_LIMB "not accounted for"
#endif

  return u.f;
}
