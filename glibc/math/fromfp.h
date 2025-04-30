/* Round to integer type.  Common helper functions.
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
#include <float.h>
#include <math.h>
#include <math-barriers.h>
#include <stdbool.h>
#include <stdint.h>

/* The including file should have defined UNSIGNED to 0 (signed return
   type) or 1 (unsigned return type), INEXACT to 0 (no inexact
   exceptions) or 1 (raise inexact exceptions) and RET_TYPE to the
   return type (intmax_t or uintmax_t).  */

/* Return the maximum unbiased exponent for an argument (negative if
   NEGATIVE is set) that might be in range for a call to a fromfp
   function with width WIDTH (greater than 0, and not exceeding that
   of intmax_t).  The truncated argument may still be out of range in
   the case of negative arguments, and if not out of range it may
   become out of range as a result of rounding.  */

static int
fromfp_max_exponent (bool negative, int width)
{
  if (UNSIGNED)
    return negative ? -1 : width - 1;
  else
    return negative ? width - 1 : width - 2;
}

/* Return the result of rounding an integer value X (passed as the
   absolute value; NEGATIVE is true if the value is negative), where
   HALF_BIT is true if the bit with value 0.5 is set and MORE_BITS is
   true if any lower bits are set, in the rounding direction
   ROUND.  */

static uintmax_t
fromfp_round (bool negative, uintmax_t x, bool half_bit, bool more_bits,
	      int round)
{
  switch (round)
    {
    case FP_INT_UPWARD:
      return x + (!negative && (half_bit || more_bits));

    case FP_INT_DOWNWARD:
      return x + (negative && (half_bit || more_bits));

    case FP_INT_TOWARDZERO:
    default:
      /* Unknown rounding directions are defined to mean unspecified
	 rounding; treat this as truncation.  */
      return x;

    case FP_INT_TONEARESTFROMZERO:
      return x + half_bit;

    case FP_INT_TONEAREST:
      return x + (half_bit && ((x & 1) || more_bits));
    }
}

/* Integer rounding, of a value whose exponent EXPONENT did not exceed
   the maximum exponent MAX_EXPONENT and so did not necessarily
   overflow, has produced X (possibly wrapping to 0); the sign is
   negative if NEGATIVE is true.  Return whether this overflowed the
   allowed width.  */

static bool
fromfp_overflowed (bool negative, uintmax_t x, int exponent, int max_exponent)
{
  if (UNSIGNED)
    {
      if (negative)
	return x != 0;
      else if (max_exponent == INTMAX_WIDTH - 1)
	return exponent == INTMAX_WIDTH - 1 && x == 0;
      else
	return x == (1ULL << (max_exponent + 1));
    }
  else
    {
      if (negative)
	return exponent == max_exponent && x != (1ULL << max_exponent);
      else
	return x == (1ULL << (max_exponent + 1));
    }
}

/* Handle a domain error for a call to a fromfp function with an
   argument which is negative if NEGATIVE is set, and specified width
   (not exceeding that of intmax_t) WIDTH.  The return value is
   unspecified (with it being unclear if the result needs to fit
   within WIDTH bits in this case); we choose to saturate to the given
   number of bits (treating NaNs like any other value).  */

static RET_TYPE
fromfp_domain_error (bool negative, unsigned int width)
{
  feraiseexcept (FE_INVALID);
  __set_errno (EDOM);
  /* The return value is unspecified; we choose to saturate to the
     given number of bits (treating NaNs like any other value).  */
  if (UNSIGNED)
    {
      if (negative)
	return 0;
      else if (width == INTMAX_WIDTH)
	return -1;
      else
	return (1ULL << width) - 1;
    }
  else
    {
      if (width == 0)
	return 0;
      else if (negative)
	return -(1ULL << (width - 1));
      else
	return (1ULL << (width - 1)) - 1;
    }
}

/* Given X, the absolute value of a floating-point number (negative if
   NEGATIVE is set) truncated towards zero, where HALF_BIT is true if
   the bit with value 0.5 is set and MORE_BITS is true if any lower
   bits are set, round it in the rounding direction ROUND, handle
   errors and exceptions and return the appropriate return value for a
   fromfp function.  X originally had floating-point exponent
   EXPONENT, which does not exceed MAX_EXPONENT, the return value from
   fromfp_max_exponent with width WIDTH.  */

static RET_TYPE
fromfp_round_and_return (bool negative, uintmax_t x, bool half_bit,
			 bool more_bits, int round, int exponent,
			 int max_exponent, unsigned int width)
{
  uintmax_t uret = fromfp_round (negative, x, half_bit, more_bits, round);
  if (fromfp_overflowed (negative, uret, exponent, max_exponent))
    return fromfp_domain_error (negative, width);

  if (INEXACT && (half_bit || more_bits))
    {
      /* There is no need for this to use the specific floating-point
	 type for which this header is included, and there is no need
	 for this header to know that type at all, so just use float
	 here.  */
      float force_inexact = 1.0f + FLT_MIN;
      math_force_eval (force_inexact);
    }
  if (UNSIGNED)
    /* A negative argument not rounding to zero will already have
       produced a domain error.  */
    return uret;
  else
    return negative ? -uret : uret;
}
