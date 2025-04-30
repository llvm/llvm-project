/* Compute remainder and a congruent to the quotient.
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

#include <math.h>

#include <math_private.h>
#include <libm-alias-double.h>
#include <stdint.h>

static const double zero = 0.0;


double
__remquo (double x, double y, int *quo)
{
  int64_t hx, hy;
  uint64_t sx, qs;
  int cquo;

  EXTRACT_WORDS64 (hx, x);
  EXTRACT_WORDS64 (hy, y);
  sx = hx & UINT64_C(0x8000000000000000);
  qs = sx ^ (hy & UINT64_C(0x8000000000000000));
  hy &= UINT64_C(0x7fffffffffffffff);
  hx &= UINT64_C(0x7fffffffffffffff);

  /* Purge off exception values.  */
  if (__glibc_unlikely (hy == 0))
    return (x * y) / (x * y);			/* y = 0 */
  if (__builtin_expect (hx >= UINT64_C(0x7ff0000000000000) /* x not finite */
			|| hy > UINT64_C(0x7ff0000000000000), 0))/* y is NaN */
    return (x * y) / (x * y);

  if (hy <= UINT64_C(0x7fbfffffffffffff))
    x = __ieee754_fmod (x, 8 * y);		/* now x < 8y */

  if (__glibc_unlikely (hx == hy))
    {
      *quo = qs ? -1 : 1;
      return zero * x;
    }

  x = fabs (x);
  INSERT_WORDS64 (y, hy);
  cquo = 0;

  if (hy <= UINT64_C(0x7fcfffffffffffff) && x >= 4 * y)
    {
      x -= 4 * y;
      cquo += 4;
    }
  if (hy <= UINT64_C(0x7fdfffffffffffff) && x >= 2 * y)
    {
      x -= 2 * y;
      cquo += 2;
    }

  if (hy < UINT64_C(0x0020000000000000))
    {
      if (x + x > y)
	{
	  x -= y;
	  ++cquo;
	  if (x + x >= y)
	    {
	      x -= y;
	      ++cquo;
	    }
	}
    }
  else
    {
      double y_half = 0.5 * y;
      if (x > y_half)
	{
	  x -= y;
	  ++cquo;
	  if (x >= y_half)
	    {
	      x -= y;
	      ++cquo;
	    }
	}
    }

  *quo = qs ? -cquo : cquo;

  /* Ensure correct sign of zero result in round-downward mode.  */
  if (x == 0.0)
    x = 0.0;
  if (sx)
    x = -x;
  return x;
}
libm_alias_double (__remquo, remquo)
