/* Round to nearest integer value, rounding halfway cases to even.
   flt-32 version.
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
#include <libm-alias-float.h>
#include <math-use-builtins.h>
#include <stdint.h>

#define BIAS 0x7f
#define MANT_DIG 24
#define MAX_EXP (2 * BIAS + 1)

float
__roundevenf (float x)
{
#if USE_ROUNDEVENF_BUILTIN
  return __builtin_roundevenf (x);
#else
  uint32_t ix, ux;
  GET_FLOAT_WORD (ix, x);
  ux = ix & 0x7fffffff;
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
      uint32_t half_bit = 1U << half_pos;
      uint32_t int_bit = 1U << int_pos;
      if ((ix & (int_bit | (half_bit - 1))) != 0)
	/* Carry into the exponent works correctly.  No need to test
	   whether HALF_BIT is set.  */
	ix += half_bit;
      ix &= ~(int_bit - 1);
    }
  else if (exponent == BIAS - 1 && ux > 0x3f000000)
    /* Interval (0.5, 1).  */
    ix = (ix & 0x80000000) | 0x3f800000;
  else
    /* Rounds to 0.  */
    ix &= 0x80000000;
  SET_FLOAT_WORD (x, ix);
  return x;
#endif /* ! USE_ROUNDEVENF_BUILTIN  */
}
#ifndef __roundevenf
libm_alias_float (__roundeven, roundeven)
#endif
