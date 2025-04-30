/* Round to nearest integer value, rounding halfway cases to even.
   ldbl-96 version.
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
#include <stdint.h>

#define BIAS 0x3fff
#define MANT_DIG 64
#define MAX_EXP (2 * BIAS + 1)

long double
__roundevenl (long double x)
{
  uint16_t se;
  uint32_t hx, lx;
  GET_LDOUBLE_WORDS (se, hx, lx, x);
  int exponent = se & 0x7fff;
  if (exponent >= BIAS + MANT_DIG - 1)
    {
      /* Integer, infinity or NaN.  */
      if (exponent == MAX_EXP)
	/* Infinity or NaN; quiet signaling NaNs.  */
	return x + x;
      else
	return x;
    }
  else if (exponent >= BIAS + MANT_DIG - 32)
    {
      /* Not necessarily an integer; integer bit is in low word.
	 Locate the bits with exponents 0 and -1.  */
      int int_pos = (BIAS + MANT_DIG - 1) - exponent;
      int half_pos = int_pos - 1;
      uint32_t half_bit = 1U << half_pos;
      uint32_t int_bit = 1U << int_pos;
      if ((lx & (int_bit | (half_bit - 1))) != 0)
	{
	  /* No need to test whether HALF_BIT is set.  */
	  lx += half_bit;
	  if (lx < half_bit)
	    {
	      hx++;
	      if (hx == 0)
		{
		  hx = 0x80000000;
		  se++;
		}
	    }
	}
      lx &= ~(int_bit - 1);
    }
  else if (exponent == BIAS + MANT_DIG - 33)
    {
      /* Not necessarily an integer; integer bit is bottom of high
	 word, half bit is top of low word.  */
      if (((hx & 1) | (lx & 0x7fffffff)) != 0)
	{
	  lx += 0x80000000;
	  if (lx < 0x80000000)
	    {
	      hx++;
	      if (hx == 0)
		{
		  hx = 0x80000000;
		  se++;
		}
	    }
	}
      lx = 0;
    }
  else if (exponent >= BIAS)
    {
      /* At least 1; not necessarily an integer, integer bit and half
	 bit are in the high word.  Locate the bits with exponents 0
	 and -1.  */
      int int_pos = (BIAS + MANT_DIG - 33) - exponent;
      int half_pos = int_pos - 1;
      uint32_t half_bit = 1U << half_pos;
      uint32_t int_bit = 1U << int_pos;
      if (((hx & (int_bit | (half_bit - 1))) | lx) != 0)
	{
	  hx += half_bit;
	  if (hx < half_bit)
	    {
	      hx = 0x80000000;
	      se++;
	    }
	}
      hx &= ~(int_bit - 1);
      lx = 0;
    }
  else if (exponent == BIAS - 1 && (hx > 0x80000000 || lx != 0))
    {
      /* Interval (0.5, 1).  */
      se = (se & 0x8000) | 0x3fff;
      hx = 0x80000000;
      lx = 0;
    }
  else
    {
      /* Rounds to 0.  */
      se &= 0x8000;
      hx = 0;
      lx = 0;
    }
  SET_LDOUBLE_WORDS (x, se, hx, lx);
  return x;
}
libm_alias_ldouble (__roundeven, roundeven)
