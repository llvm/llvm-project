/* Round to integer type.  ldbl-128 version.
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
#define MANT_DIG 113

#if UNSIGNED
# define RET_TYPE uintmax_t
#else
# define RET_TYPE intmax_t
#endif

#include <fromfp.h>

RET_TYPE
FUNC (_Float128 x, int round, unsigned int width)
{
  if (width > INTMAX_WIDTH)
    width = INTMAX_WIDTH;
  uint64_t hx, lx;
  GET_LDOUBLE_WORDS64 (hx, lx, x);
  bool negative = (hx & 0x8000000000000000ULL) != 0;
  if (width == 0)
    return fromfp_domain_error (negative, width);
  hx &= 0x7fffffffffffffffULL;
  if ((hx | lx) == 0)
    return 0;
  int exponent = hx >> (MANT_DIG - 1 - 64);
  exponent -= BIAS;
  int max_exponent = fromfp_max_exponent (negative, width);
  if (exponent > max_exponent)
    return fromfp_domain_error (negative, width);

  hx &= ((1ULL << (MANT_DIG - 1 - 64)) - 1);
  hx |= 1ULL << (MANT_DIG - 1 - 64);
  uintmax_t uret;
  bool half_bit, more_bits;
  /* The exponent is at most 63, so we are shifting right by at least
     49 bits.  */
  if (exponent >= -1)
    {
      int shift = MANT_DIG - 1 - exponent;
      if (shift <= 64)
	{
	  uint64_t h = 1ULL << (shift - 1);
	  half_bit = (lx & h) != 0;
	  more_bits = (lx & (h - 1)) != 0;
	  uret = hx << (64 - shift);
	  if (shift != 64)
	    uret |= lx >> shift;
	}
      else
	{
	  uint64_t h = 1ULL << (shift - 1 - 64);
	  half_bit = (hx & h) != 0;
	  more_bits = ((hx & (h - 1)) | lx) != 0;
	  uret = hx >> (shift - 64);
	}
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
