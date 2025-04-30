/* Round long double to integer away from zero.
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

#define NO_MATH_REDIRECT
#include <math.h>

#include <math_private.h>
#include <libm-alias-ldouble.h>
#include <math-use-builtins.h>


_Float128
__roundl (_Float128 x)
{
#if USE_ROUNDL_BUILTIN
  return __builtin_roundl (x);
#else
  /* Use generic implementation.  */
  int32_t j0;
  uint64_t i1, i0;

  GET_LDOUBLE_WORDS64 (i0, i1, x);
  j0 = ((i0 >> 48) & 0x7fff) - 0x3fff;
  if (j0 < 48)
    {
      if (j0 < 0)
	{
	  i0 &= 0x8000000000000000ULL;
	  if (j0 == -1)
	    i0 |= 0x3fff000000000000LL;
	  i1 = 0;
	}
      else
	{
	  uint64_t i = 0x0000ffffffffffffLL >> j0;
	  if (((i0 & i) | i1) == 0)
	    /* X is integral.  */
	    return x;

	  i0 += 0x0000800000000000LL >> j0;
	  i0 &= ~i;
	  i1 = 0;
	}
    }
  else if (j0 > 111)
    {
      if (j0 == 0x4000)
	/* Inf or NaN.  */
	return x + x;
      else
	return x;
    }
  else
    {
      uint64_t i = -1ULL >> (j0 - 48);
      if ((i1 & i) == 0)
	/* X is integral.  */
	return x;

      uint64_t j = i1 + (1LL << (111 - j0));
      if (j < i1)
	i0 += 1;
      i1 = j;
      i1 &= ~i;
    }

  SET_LDOUBLE_WORDS64 (x, i0, i1);
  return x;
#endif /* ! USE_ROUNDL_BUILTIN  */
}
libm_alias_ldouble (__round, round)
