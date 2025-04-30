/* Round to nearest integer value, rounding halfway cases to even.
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
#include <libm-alias-double.h>
#include <stdint.h>
#include <math-use-builtins.h>

#define BIAS 0x3ff
#define MANT_DIG 53
#define MAX_EXP (2 * BIAS + 1)

double
__roundeven (double x)
{
#if USE_ROUNDEVEN_BUILTIN
  return __builtin_roundeven (x);
#else
  uint64_t ix, ux;
  EXTRACT_WORDS64 (ix, x);
  ux = ix & 0x7fffffffffffffffULL;
  int exponent = ux >> (MANT_DIG - 1);
  if (exponent >= BIAS + MANT_DIG - 1)
    {
      /* Integer, infinity or NaN.  */
      if (exponent == MAX_EXP)
	/* Infinity or NaN; quiet signaling NaNs.  */
	return x + x;
      else
	return x;
    }
  else if (exponent >= BIAS)
    {
      /* At least 1; not necessarily an integer.  Locate the bits with
	 exponents 0 and -1 (when the unbiased exponent is 0, the bit
	 with exponent 0 is implicit, but as the bias is odd it is OK
	 to take it from the low bit of the exponent).  */
      int int_pos = (BIAS + MANT_DIG - 1) - exponent;
      int half_pos = int_pos - 1;
      uint64_t half_bit = 1ULL << half_pos;
      uint64_t int_bit = 1ULL << int_pos;
      if ((ix & (int_bit | (half_bit - 1))) != 0)
	/* Carry into the exponent works correctly.  No need to test
	   whether HALF_BIT is set.  */
	ix += half_bit;
      ix &= ~(int_bit - 1);
    }
  else if (exponent == BIAS - 1 && ux > 0x3fe0000000000000ULL)
    /* Interval (0.5, 1).  */
    ix = (ix & 0x8000000000000000ULL) | 0x3ff0000000000000ULL;
  else
    /* Rounds to 0.  */
    ix &= 0x8000000000000000ULL;
  INSERT_WORDS64 (x, ix);
  return x;
#endif /* ! USE_ROUNDEVEN_BUILTIN  */
}
#ifndef __roundeven
libm_alias_double (__roundeven, roundeven)
#endif
