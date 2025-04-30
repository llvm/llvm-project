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


static const float zero = 0.0;
#include <libm-alias-float.h>


float
__remquof (float x, float y, int *quo)
{
  int32_t hx,hy;
  uint32_t sx;
  int cquo, qs;

  GET_FLOAT_WORD (hx, x);
  GET_FLOAT_WORD (hy, y);
  sx = hx & 0x80000000;
  qs = sx ^ (hy & 0x80000000);
  hy &= 0x7fffffff;
  hx &= 0x7fffffff;

  /* Purge off exception values.  */
  if (hy == 0)
    return (x * y) / (x * y); 			/* y = 0 */
  if ((hx >= 0x7f800000)			/* x not finite */
      || (hy > 0x7f800000))			/* y is NaN */
    return (x * y) / (x * y);

  if (hy <= 0x7dffffff)
    x = __ieee754_fmodf (x, 8 * y);		/* now x < 8y */

  if ((hx - hy) == 0)
    {
      *quo = qs ? -1 : 1;
      return zero * x;
    }

  x  = fabsf (x);
  y  = fabsf (y);
  cquo = 0;

  if (hy <= 0x7e7fffff && x >= 4 * y)
    {
      x -= 4 * y;
      cquo += 4;
    }
  if (hy <= 0x7effffff && x >= 2 * y)
    {
      x -= 2 * y;
      cquo += 2;
    }

  if (hy < 0x01000000)
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
      float y_half = 0.5 * y;
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
  if (x == 0.0f)
    x = 0.0f;
  if (sx)
    x = -x;
  return x;
}
libm_alias_float (__remquo, remquo)
