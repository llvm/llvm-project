/* Round float to integer away from zero.
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
#include <libm-alias-float.h>
#include <math-use-builtins.h>


float
__roundf (float x)
{
#if USE_ROUNDF_BUILTIN
  return __builtin_roundf (x);
#else
  /* Use generic implementation.  */
  int32_t i0, j0;

  GET_FLOAT_WORD (i0, x);
  j0 = ((i0 >> 23) & 0xff) - 0x7f;
  if (j0 < 23)
    {
      if (j0 < 0)
	{
	  i0 &= 0x80000000;
	  if (j0 == -1)
	    i0 |= 0x3f800000;
	}
      else
	{
	  uint32_t i = 0x007fffff >> j0;
	  if ((i0 & i) == 0)
	    /* X is integral.  */
	    return x;

	  i0 += 0x00400000 >> j0;
	  i0 &= ~i;
	}
    }
  else
    {
      if (j0 == 0x80)
	/* Inf or NaN.  */
	return x + x;
      else
	return x;
    }

  SET_FLOAT_WORD (x, i0);
  return x;
#endif /* ! USE_ROUNDF_BUILTIN  */
}
libm_alias_float (__round, round)
