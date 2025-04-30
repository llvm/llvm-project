/* Round argument to nearest integral value according to current rounding
   direction.
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

static const _Float128 two112[2] =
{
  L(5.19229685853482762853049632922009600E+33), /* 0x406F000000000000, 0 */
 L(-5.19229685853482762853049632922009600E+33)  /* 0xC06F000000000000, 0 */
};

long int
__lrintl (_Float128 x)
{
  int32_t j0;
  uint64_t i0,i1;
  _Float128 w;
  _Float128 t;
  long int result;
  int sx;

  GET_LDOUBLE_WORDS64 (i0, i1, x);
  j0 = ((i0 >> 48) & 0x7fff) - 0x3fff;
  sx = i0 >> 63;
  i0 &= 0x0000ffffffffffffLL;
  i0 |= 0x0001000000000000LL;

  if (j0 < (int32_t) (8 * sizeof (long int)) - 1)
    {
      if (j0 < 48)
	{
#if defined FE_INVALID || defined FE_INEXACT
	  /* X < LONG_MAX + 1 implied by J0 < 31.  */
	  if (sizeof (long int) == 4
	      && x > (_Float128) LONG_MAX)
	    {
	      /* In the event of overflow we must raise the "invalid"
		 exception, but not "inexact".  */
	      t = __nearbyintl (x);
	      feraiseexcept (t == LONG_MAX ? FE_INEXACT : FE_INVALID);
	    }
	  else
#endif
	    {
	      w = two112[sx] + x;
	      t = w - two112[sx];
	    }
	  GET_LDOUBLE_WORDS64 (i0, i1, t);
	  j0 = ((i0 >> 48) & 0x7fff) - 0x3fff;
	  i0 &= 0x0000ffffffffffffLL;
	  i0 |= 0x0001000000000000LL;

	  result = (j0 < 0 ? 0 : i0 >> (48 - j0));
	}
      else if (j0 >= 112)
	result = ((long int) i0 << (j0 - 48)) | (i1 << (j0 - 112));
      else
	{
#if defined FE_INVALID || defined FE_INEXACT
	  /* X < LONG_MAX + 1 implied by J0 < 63.  */
	  if (sizeof (long int) == 8
	      && x > (_Float128) LONG_MAX)
	    {
	      /* In the event of overflow we must raise the "invalid"
		 exception, but not "inexact".  */
	      t = __nearbyintl (x);
	      feraiseexcept (t == LONG_MAX ? FE_INEXACT : FE_INVALID);
	    }
	  else
#endif
	    {
	      w = two112[sx] + x;
	      t = w - two112[sx];
	    }
	  GET_LDOUBLE_WORDS64 (i0, i1, t);
	  j0 = ((i0 >> 48) & 0x7fff) - 0x3fff;
	  i0 &= 0x0000ffffffffffffLL;
	  i0 |= 0x0001000000000000LL;

	  if (j0 == 48)
	    result = (long int) i0;
	  else
	    result = ((long int) i0 << (j0 - 48)) | (i1 >> (112 - j0));
	}
    }
  else
    {
      /* The number is too large.  Unless it rounds to LONG_MIN,
	 FE_INVALID must be raised and the return value is
	 unspecified.  */
#if defined FE_INVALID || defined FE_INEXACT
      if (x < (_Float128) LONG_MIN
	  && x > (_Float128) LONG_MIN - 1)
	{
	  /* If truncation produces LONG_MIN, the cast will not raise
	     the exception, but may raise "inexact".  */
	  t = __nearbyintl (x);
	  feraiseexcept (t == LONG_MIN ? FE_INEXACT : FE_INVALID);
	  return LONG_MIN;
	}
      else if (FIX_LDBL_LONG_CONVERT_OVERFLOW && x != (_Float128) LONG_MIN)
	{
	  feraiseexcept (FE_INVALID);
	  return sx == 0 ? LONG_MAX : LONG_MIN;
	}

#endif
      return (long int) x;
    }

  return sx ? -result : result;
}

libm_alias_ldouble (__lrint, lrint)
