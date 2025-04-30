/* Round to integer type.  ldbl-128ibm version.
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
#include <stdbool.h>
#include <stdint.h>

#define BIAS 0x3ff
#define MANT_DIG 53

#if UNSIGNED
# define RET_TYPE uintmax_t
#else
# define RET_TYPE intmax_t
#endif

#include <fromfp.h>

RET_TYPE
FUNC (long double x, int round, unsigned int width)
{
  double hi, lo;
  if (width > INTMAX_WIDTH)
    width = INTMAX_WIDTH;
  uint64_t hx, lx;
  ldbl_unpack (x, &hi, &lo);
  EXTRACT_WORDS64 (hx, hi);
  EXTRACT_WORDS64 (lx, lo);
  bool negative = (hx & 0x8000000000000000ULL) != 0;
  bool lo_negative = (lx & 0x8000000000000000ULL) != 0;
  if (width == 0)
    return fromfp_domain_error (negative, width);
  hx &= 0x7fffffffffffffffULL;
  lx &= 0x7fffffffffffffffULL;
  if ((hx | lx) == 0)
    return 0;
  int hi_exponent = hx >> (MANT_DIG - 1);
  hi_exponent -= BIAS;
  int exponent = hi_exponent;
  hx &= ((1ULL << (MANT_DIG - 1)) - 1);
  if (hx == 0 && lx != 0 && lo_negative != negative)
    exponent--;
  int max_exponent = fromfp_max_exponent (negative, width);
  if (exponent > max_exponent)
    return fromfp_domain_error (negative, width);
  int lo_exponent = lx >> (MANT_DIG - 1);
  lo_exponent -= BIAS;

  /* Convert the high part to integer.  */
  hx |= 1ULL << (MANT_DIG - 1);
  uintmax_t uret;
  bool half_bit, more_bits;
  if (hi_exponent >= MANT_DIG - 1)
    {
      uret = hx;
      uret <<= hi_exponent - (MANT_DIG - 1);
      half_bit = false;
      more_bits = false;
    }
  else if (hi_exponent >= -1)
    {
      uint64_t h = 1ULL << (MANT_DIG - 2 - hi_exponent);
      half_bit = (hx & h) != 0;
      more_bits = (hx & (h - 1)) != 0;
      uret = hx >> (MANT_DIG - 1 - hi_exponent);
    }
  else
    {
      uret = 0;
      half_bit = false;
      more_bits = true;
    }

  /* Likewise, the low part.  */
  if (lx != 0)
    {
      uintmax_t lo_uret;
      bool lo_half_bit, lo_more_bits;
      lx &= ((1ULL << (MANT_DIG - 1)) - 1);
      lx |= 1ULL << (MANT_DIG - 1);
      /* The high part exponent is at most 64, so the low part
	 exponent is at most 11.  */
      if (lo_exponent >= -1)
	{
	  uint64_t h = 1ULL << (MANT_DIG - 2 - lo_exponent);
	  lo_half_bit = (lx & h) != 0;
	  lo_more_bits = (lx & (h - 1)) != 0;
	  lo_uret = lx >> (MANT_DIG - 1 - lo_exponent);
	}
      else
	{
	  lo_uret = 0;
	  lo_half_bit = false;
	  lo_more_bits = true;
	}
      if (lo_negative == negative)
	{
	  uret += lo_uret;
	  half_bit |= lo_half_bit;
	  more_bits |= lo_more_bits;
	}
      else
	{
	  uret -= lo_uret;
	  if (lo_half_bit)
	    {
	      uret--;
	      half_bit = true;
	    }
	  if (lo_more_bits && !more_bits)
	    {
	      if (half_bit)
		{
		  half_bit = false;
		  more_bits = true;
		}
	      else
		{
		  uret--;
		  half_bit = true;
		  more_bits = true;
		}
	    }
	}
    }

  return fromfp_round_and_return (negative, uret, half_bit, more_bits, round,
				  exponent, max_exponent, width);
}
