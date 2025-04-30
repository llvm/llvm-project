/* Round to integer type.  ldbl-96 version.
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

#include <errno.h>
#include <fenv.h>
#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>
#include <stdbool.h>
#include <stdint.h>

#define BIAS 0x3fff
#define MANT_DIG 64

#if UNSIGNED
# define RET_TYPE uintmax_t
#else
# define RET_TYPE intmax_t
#endif

#include <fromfp.h>

RET_TYPE
FUNC (long double x, int round, unsigned int width)
{
  if (width > INTMAX_WIDTH)
    width = INTMAX_WIDTH;
  uint16_t se;
  uint32_t hx, lx;
  GET_LDOUBLE_WORDS (se, hx, lx, x);
  bool negative = (se & 0x8000) != 0;
  if (width == 0)
    return fromfp_domain_error (negative, width);
  if ((hx | lx) == 0)
    return 0;
  int exponent = se & 0x7fff;
  exponent -= BIAS;
  int max_exponent = fromfp_max_exponent (negative, width);
  if (exponent > max_exponent)
    return fromfp_domain_error (negative, width);

  uint64_t ix = (((uint64_t) hx) << 32) | lx;
  uintmax_t uret;
  bool half_bit, more_bits;
  if (exponent >= MANT_DIG - 1)
    {
      uret = ix;
      /* Exponent 63; no shifting required.  */
      half_bit = false;
      more_bits = false;
    }
  else if (exponent >= -1)
    {
      uint64_t h = 1ULL << (MANT_DIG - 2 - exponent);
      half_bit = (ix & h) != 0;
      more_bits = (ix & (h - 1)) != 0;
      if (exponent == -1)
	uret = 0;
      else
	uret = ix >> (MANT_DIG - 1 - exponent);
    }
  else
    {
      uret = 0;
      half_bit = false;
      more_bits = true;
    }
  return fromfp_round_and_return (negative, uret, half_bit, more_bits, round,
				  exponent, max_exponent, width);
}
