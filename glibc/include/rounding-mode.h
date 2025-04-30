/* Handle floating-point rounding mode within libc.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
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

#ifndef _ROUNDING_MODE_H
#define _ROUNDING_MODE_H	1

#include <fenv.h>
#include <stdbool.h>
#include <stdlib.h>

/* Get the architecture-specific definition of how to determine the
   rounding mode in libc.  This header must also define the FE_*
   macros for any standard rounding modes the architecture does not
   have in <fenv.h>, to arbitrary distinct values.  */
#include <get-rounding-mode.h>

/* Return true if a number should be rounded away from zero in
   rounding mode MODE, false otherwise.  NEGATIVE is true if the
   number is negative, false otherwise.  LAST_DIGIT_ODD is true if the
   last digit of the truncated value (last bit for binary) is odd,
   false otherwise.  HALF_BIT is true if the number is at least half
   way from the truncated value to the next value with the
   least-significant digit in the same place, false otherwise.
   MORE_BITS is true if the number is not exactly equal to the
   truncated value or the half-way value, false otherwise.  */

static bool
round_away (bool negative, bool last_digit_odd, bool half_bit, bool more_bits,
	    int mode)
{
  switch (mode)
    {
    case FE_DOWNWARD:
      return negative && (half_bit || more_bits);

    case FE_TONEAREST:
      return half_bit && (last_digit_odd || more_bits);

    case FE_TOWARDZERO:
      return false;

    case FE_UPWARD:
      return !negative && (half_bit || more_bits);

    default:
      abort ();
    }
}

#endif /* rounding-mode.h */
