/* Compute remainder and a congruent to the quotient.
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

#include <math.h>

#include <math_private.h>
#include <libm-alias-ldouble.h>


static const _Float128 zero = 0.0;


_Float128
__remquol (_Float128 x, _Float128 y, int *quo)
{
  int64_t hx,hy;
  uint64_t sx,lx,ly,qs;
  int cquo;

  GET_LDOUBLE_WORDS64 (hx, lx, x);
  GET_LDOUBLE_WORDS64 (hy, ly, y);
  sx = hx & 0x8000000000000000ULL;
  qs = sx ^ (hy & 0x8000000000000000ULL);
  hy &= 0x7fffffffffffffffLL;
  hx &= 0x7fffffffffffffffLL;

  /* Purge off exception values.  */
  if ((hy | ly) == 0)
    return (x * y) / (x * y); 			/* y = 0 */
  if ((hx >= 0x7fff000000000000LL)		/* x not finite */
      || ((hy >= 0x7fff000000000000LL)		/* y is NaN */
	  && (((hy - 0x7fff000000000000LL) | ly) != 0)))
    return (x * y) / (x * y);

  if (hy <= 0x7ffbffffffffffffLL)
    x = __ieee754_fmodl (x, 8 * y);              /* now x < 8y */

  if (((hx - hy) | (lx - ly)) == 0)
    {
      *quo = qs ? -1 : 1;
      return zero * x;
    }

  x  = fabsl (x);
  y  = fabsl (y);
  cquo = 0;

  if (hy <= 0x7ffcffffffffffffLL && x >= 4 * y)
    {
      x -= 4 * y;
      cquo += 4;
    }
  if (hy <= 0x7ffdffffffffffffLL && x >= 2 * y)
    {
      x -= 2 * y;
      cquo += 2;
    }

  if (hy < 0x0002000000000000LL)
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
      _Float128 y_half = L(0.5) * y;
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
  if (x == 0)
    x = 0;
  if (sx)
    x = -x;
  return x;
}
libm_alias_ldouble (__remquo, remquo)
