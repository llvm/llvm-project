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


long long int
__llrintl (long double x)
{
  int32_t se,j0;
  uint32_t i0, i1;
  long long int result;
  long double w;
  long double t;
  int sx;

  GET_LDOUBLE_WORDS (se, i0, i1, x);

  sx = (se >> 15) & 1;
  j0 = (se & 0x7fff) - 0x3fff;

  if (j0 < (int32_t) (8 * sizeof (long long int)) - 1)
    {
      if (j0 >= 63)
	result = (((long long int) i0 << 32) | i1) << (j0 - 63);
      else
	{
#if defined FE_INVALID || defined FE_INEXACT
	  /* X < LLONG_MAX + 1 implied by J0 < 63.  */
	  if (x > (long double) LLONG_MAX)
	    {
	      /* In the event of overflow we must raise the "invalid"
		 exception, but not "inexact".  */
	      t = __nearbyintl (x);
	      feraiseexcept (t == LLONG_MAX ? FE_INEXACT : FE_INVALID);
	    }
	  else
#endif
	    {
	      w = two63[sx] + x;
	      t = w - two63[sx];
	    }
	  GET_LDOUBLE_WORDS (se, i0, i1, t);
	  j0 = (se & 0x7fff) - 0x3fff;

	  if (j0 < 0)
	    result = 0;
	  else if (j0 <= 31)
	    result = i0 >> (31 - j0);
	  else
	    result = ((long long int) i0 << (j0 - 31)) | (i1 >> (63 - j0));
	}
    }
  else
    {
      /* The number is too large.  It is left implementation defined
	 what happens.  */
      return (long long int) x;
    }

  return sx ? -result : result;
}

libm_alias_ldouble (__llrint, llrint)
