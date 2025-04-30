/* Round argument to nearest integral value according to current rounding
   direction.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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

static const long double two63[2] =
{
  9.223372036854775808000000e+18, /* 0x403E, 0x00000000, 0x00000000 */
 -9.223372036854775808000000e+18  /* 0xC03E, 0x00000000, 0x00000000 */
};


long int
__lrintl (long double x)
{
  int32_t se,j0;
  uint32_t i0, i1;
  long int result;
  long double w;
  long double t;
  int sx;

  GET_LDOUBLE_WORDS (se, i0, i1, x);

  sx = (se >> 15) & 1;
  j0 = (se & 0x7fff) - 0x3fff;

  if (j0 < 31)
    {
#if defined FE_INVALID || defined FE_INEXACT
      /* X < LONG_MAX + 1 implied by J0 < 31.  */
      if (sizeof (long int) == 4
	  && x > (long double) LONG_MAX)
	{
	  /* In the event of overflow we must raise the "invalid"
	     exception, but not "inexact".  */
	  t = __nearbyintl (x);
	  feraiseexcept (t == LONG_MAX ? FE_INEXACT : FE_INVALID);
	}
      else
#endif
	{
	  w = two63[sx] + x;
	  t = w - two63[sx];
	}
      GET_LDOUBLE_WORDS (se, i0, i1, t);
      j0 = (se & 0x7fff) - 0x3fff;

      result = (j0 < 0 ? 0 : i0 >> (31 - j0));
    }
  else if (j0 < (int32_t) (8 * sizeof (long int)) - 1)
    {
      if (j0 >= 63)
	result = ((long int) i0 << (j0 - 31)) | (i1 << (j0 - 63));
      else
	{
#if defined FE_INVALID || defined FE_INEXACT
	  /* X < LONG_MAX + 1 implied by J0 < 63.  */
	  if (sizeof (long int) == 8
	      && x > (long double) LONG_MAX)
	    {
	      /* In the event of overflow we must raise the "invalid"
		 exception, but not "inexact".  */
	      t = __nearbyintl (x);
	      feraiseexcept (t == LONG_MAX ? FE_INEXACT : FE_INVALID);
	    }
	  else
#endif
	    {
	      w = two63[sx] + x;
	      t = w - two63[sx];
	    }
	  GET_LDOUBLE_WORDS (se, i0, i1, t);
	  j0 = (se & 0x7fff) - 0x3fff;

	  if (j0 == 31)
	    result = (long int) i0;
	  else
	    result = ((long int) i0 << (j0 - 31)) | (i1 >> (63 - j0));
	}
    }
  else
    {
      /* The number is too large.  Unless it rounds to LONG_MIN,
	 FE_INVALID must be raised and the return value is
	 unspecified.  */
#if defined FE_INVALID || defined FE_INEXACT
      if (sizeof (long int) == 4
	  && x < (long double) LONG_MIN
	  && x > (long double) LONG_MIN - 1.0L)
	{
	  /* If truncation produces LONG_MIN, the cast will not raise
	     the exception, but may raise "inexact".  */
	  t = __nearbyintl (x);
	  feraiseexcept (t == LONG_MIN ? FE_INEXACT : FE_INVALID);
	  return LONG_MIN;
	}
#endif
      return (long int) x;
    }

  return sx ? -result : result;
}

libm_alias_ldouble (__lrint, lrint)
