/* Round long double to integer away from zero.
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

#define NO_MATH_REDIRECT
#include <math.h>

#include <math_private.h>
#include <libm-alias-ldouble.h>


long double
__roundl (long double x)
{
  int32_t j0;
  uint32_t se, i1, i0;

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  j0 = (se & 0x7fff) - 0x3fff;
  if (j0 < 31)
    {
      if (j0 < 0)
	{
	  se &= 0x8000;
	  i0 = i1 = 0;
	  if (j0 == -1)
	    {
	      se |= 0x3fff;
	      i0 = 0x80000000;
	    }
	}
      else
	{
	  uint32_t i = 0x7fffffff >> j0;
	  if (((i0 & i) | i1) == 0)
	    /* X is integral.  */
	    return x;

	  uint32_t j = i0 + (0x40000000 >> j0);
	  if (j < i0)
	    se += 1;
	  i0 = (j & ~i) | 0x80000000;
	  i1 = 0;
	}
    }
  else if (j0 > 62)
    {
      if (j0 == 0x4000)
	/* Inf or NaN.  */
	return x + x;
      else
	return x;
    }
  else
    {
      uint32_t i = 0xffffffff >> (j0 - 31);
      if ((i1 & i) == 0)
	/* X is integral.  */
	return x;

      uint32_t j = i1 + (1 << (62 - j0));
      if (j < i1)
	{
	  uint32_t k = i0 + 1;
	  if (k < i0)
	    {
	      se += 1;
	      k |= 0x80000000;
	    }
	  i0 = k;
	}
      i1 = j;
      i1 &= ~i;
    }

  SET_LDOUBLE_WORDS (x, se, i0, i1);
  return x;
}
libm_alias_ldouble (__round, round)
