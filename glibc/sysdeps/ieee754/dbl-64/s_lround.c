/* Round double value to long int.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <fenv.h>
#include <limits.h>
#include <math.h>

#include <math_private.h>
#include <libm-alias-double.h>
#include <fix-fp-int-convert-overflow.h>

/* For LP64, lround is an alias for llround.  */
#ifndef _LP64

long int
__lround (double x)
{
  int32_t j0;
  int64_t i0;
  long int result;
  int sign;

  EXTRACT_WORDS64 (i0, x);
  j0 = ((i0 >> 52) & 0x7ff) - 0x3ff;
  sign = i0 < 0 ? -1 : 1;
  i0 &= UINT64_C(0xfffffffffffff);
  i0 |= UINT64_C(0x10000000000000);

  if (j0 < (int32_t) (8 * sizeof (long int)) - 1)
    {
      if (j0 < 0)
	return j0 < -1 ? 0 : sign;
      else if (j0 >= 52)
	result = i0 << (j0 - 52);
      else
	{
	  i0 += UINT64_C(0x8000000000000) >> j0;

	  result = i0 >> (52 - j0);
#ifdef FE_INVALID
	  if (sizeof (long int) == 4
	      && sign == 1
	      && result == LONG_MIN)
	    /* Rounding brought the value out of range.  */
	    feraiseexcept (FE_INVALID);
#endif
	}
    }
  else
    {
      /* The number is too large.  Unless it rounds to LONG_MIN,
	 FE_INVALID must be raised and the return value is
	 unspecified.  */
#ifdef FE_INVALID
      if (FIX_DBL_LONG_CONVERT_OVERFLOW
	  && !(sign == -1
	       && (sizeof (long int) == 4
		   ? x > (double) LONG_MIN - 0.5
		   : x >= (double) LONG_MIN)))
	{
	  feraiseexcept (FE_INVALID);
	  return sign == 1 ? LONG_MAX : LONG_MIN;
	}
      else if (!FIX_DBL_LONG_CONVERT_OVERFLOW
	  && sizeof (long int) == 4
	  && x <= (double) LONG_MIN - 0.5)
	{
	  /* If truncation produces LONG_MIN, the cast will not raise
	     the exception, but may raise "inexact".  */
	  feraiseexcept (FE_INVALID);
	  return LONG_MIN;
	}
#endif
      return (long int) x;
    }

  return sign * result;
}

libm_alias_double (__lround, lround)

#endif
