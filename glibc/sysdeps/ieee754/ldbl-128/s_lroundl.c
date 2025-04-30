/* Round long double value to long int.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997 and
		  Jakub Jelinek <jj@ultra.linux.cz>, 1999.

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
#include <libm-alias-ldouble.h>
#include <fix-fp-int-convert-overflow.h>

long int
__lroundl (_Float128 x)
{
  int64_t j0;
  uint64_t i1, i0;
  long int result;
  int sign;

  GET_LDOUBLE_WORDS64 (i0, i1, x);
  j0 = ((i0 >> 48) & 0x7fff) - 0x3fff;
  sign = (i0 & 0x8000000000000000ULL) != 0 ? -1 : 1;
  i0 &= 0x0000ffffffffffffLL;
  i0 |= 0x0001000000000000LL;

  if (j0 < (int32_t) (8 * sizeof (long int)) - 1)
    {
      if (j0 < 48)
	{
	  if (j0 < 0)
	    return j0 < -1 ? 0 : sign;
	  else
	    {
	      i0 += 0x0000800000000000LL >> j0;
	      result = i0 >> (48 - j0);
#ifdef FE_INVALID
	      if (sizeof (long int) == 4
		  && sign == 1
		  && result == LONG_MIN)
		/* Rounding brought the value out of range.  */
		feraiseexcept (FE_INVALID);
#endif
	    }
	}
      else if (j0 >= 112)
	result = ((long int) i0 << (j0 - 48)) | (i1 << (j0 - 112));
      else
	{
	  uint64_t j = i1 + (0x8000000000000000ULL >> (j0 - 48));
	  if (j < i1)
	    ++i0;

	  if (j0 == 48)
	    result = (long int) i0;
	  else
	    {
	      result = ((long int) i0 << (j0 - 48)) | (j >> (112 - j0));
#ifdef FE_INVALID
	      if (sizeof (long int) == 8
		  && sign == 1
		  && result == LONG_MIN)
		/* Rounding brought the value out of range.  */
		feraiseexcept (FE_INVALID);
#endif
	    }
	}
    }
  else
    {
      /* The number is too large.  Unless it rounds to LONG_MIN,
	 FE_INVALID must be raised and the return value is
	 unspecified.  */
#ifdef FE_INVALID
      if (FIX_LDBL_LONG_CONVERT_OVERFLOW
	  && !(sign == -1 && x > (_Float128) LONG_MIN - L(0.5)))
	{
	  feraiseexcept (FE_INVALID);
	  return sign == 1 ? LONG_MAX : LONG_MIN;
	}
      else if (!FIX_LDBL_LONG_CONVERT_OVERFLOW
	       && x <= (_Float128) LONG_MIN - L(0.5))
	{
	  /* If truncation produces LONG_MIN, the cast will not raise
	     the exception, but may raise "inexact".  */
	  feraiseexcept (FE_INVALID);
	  return LONG_MIN;
	}
#endif
      /* The number is too large.  It is left implementation defined
	 what happens.  */
      return (long int) x;
    }

  return sign * result;
}

libm_alias_ldouble (__lround, lround)
