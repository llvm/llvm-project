/* Round to nearest integer value, rounding halfway cases to even.
   ldbl-128 version.
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

#define NO_MATH_REDIRECT
#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>
#include <math-use-builtins.h>
#include <stdint.h>

#define BIAS 0x3fff
#define MANT_DIG 113
#define MAX_EXP (2 * BIAS + 1)

_Float128
__roundevenl (_Float128 x)
{
#if USE_ROUNDEVENL_BUILTIN
  return __builtin_roundevenl (x);
#else
  uint64_t hx, lx, uhx;
  GET_LDOUBLE_WORDS64 (hx, lx, x);
  uhx = hx & 0x7fffffffffffffffULL;
  int exponent = uhx >> (MANT_DIG - 1 - 64);
  if (exponent >= BIAS + MANT_DIG - 1)
    {
      /* Integer, infinity or NaN.  */
      if (exponent == MAX_EXP)
	/* Infinity or NaN; quiet signaling NaNs.  */
	return x + x;
      else
	return x;
    }
  else if (exponent >= BIAS + MANT_DIG - 64)
    {
      /* Not necessarily an integer; integer bit is in low word.
	 Locate the bits with exponents 0 and -1.  */
      int int_pos = (BIAS + MANT_DIG - 1) - exponent;
      int half_pos = int_pos - 1;
      uint64_t half_bit = 1ULL << half_pos;
      uint64_t int_bit = 1ULL << int_pos;
      if ((lx & (int_bit | (half_bit - 1))) != 0)
	{
	  /* Carry into the exponent works correctly.  No need to test
	     whether HALF_BIT is set.  */
	  lx += half_bit;
	  hx += lx < half_bit;
	}
      lx &= ~(int_bit - 1);
    }
  else if (exponent == BIAS + MANT_DIG - 65)
    {
      /* Not necessarily an integer; integer bit is bottom of high
	 word, half bit is top of low word.  */
      if (((hx & 1) | (lx & 0x7fffffffffffffffULL)) != 0)
	{
	  lx += 0x8000000000000000ULL;
	  hx += lx < 0x8000000000000000ULL;
	}
      lx = 0;
    }
  else if (exponent >= BIAS)
    {
      /* At least 1; not necessarily an integer, integer bit and half
	 bit are in the high word.  Locate the bits with exponents 0
	 and -1 (when the unbiased exponent is 0, the bit with
	 exponent 0 is implicit, but as the bias is odd it is OK to
	 take it from the low bit of the exponent).  */
      int int_pos = (BIAS + MANT_DIG - 65) - exponent;
      int half_pos = int_pos - 1;
      uint64_t half_bit = 1ULL << half_pos;
      uint64_t int_bit = 1ULL << int_pos;
      if (((hx & (int_bit | (half_bit - 1))) | lx) != 0)
	hx += half_bit;
      hx &= ~(int_bit - 1);
      lx = 0;
    }
  else if (exponent == BIAS - 1 && (uhx > 0x3ffe000000000000ULL || lx != 0))
    {
      /* Interval (0.5, 1).  */
      hx = (hx & 0x8000000000000000ULL) | 0x3fff000000000000ULL;
      lx = 0;
    }
  else
    {
      /* Rounds to 0.  */
      hx &= 0x8000000000000000ULL;
      lx = 0;
    }
  SET_LDOUBLE_WORDS64 (x, hx, lx);
  return x;
#endif /* ! USE_ROUNDEVENL_BUILTIN  */
}
libm_alias_ldouble (__roundeven, roundeven)
