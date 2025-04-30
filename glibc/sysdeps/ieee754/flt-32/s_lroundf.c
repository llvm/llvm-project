/* Round float value to long int.
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
#include <libm-alias-float.h>
#include <fix-fp-int-convert-overflow.h>


long int
__lroundf (float x)
{
  int32_t j0;
  uint32_t i;
  long int result;
  int sign;

  GET_FLOAT_WORD (i, x);
  j0 = ((i >> 23) & 0xff) - 0x7f;
  sign = (i & 0x80000000) != 0 ? -1 : 1;
  i &= 0x7fffff;
  i |= 0x800000;

  if (j0 < (int32_t) (8 * sizeof (long int)) - 1)
    {
      if (j0 < 0)
	return j0 < -1 ? 0 : sign;
      else if (j0 >= 23)
	result = (long int) i << (j0 - 23);
      else
	{
	  i += 0x400000 >> j0;

	  result = i >> (23 - j0);
	}
    }
  else
    {
#ifdef FE_INVALID
      /* The number is too large.  Unless it rounds to LONG_MIN,
	 FE_INVALID must be raised and the return value is
	 unspecified.  */
      if (FIX_FLT_LONG_CONVERT_OVERFLOW && x != (float) LONG_MIN)
	{
	  feraiseexcept (FE_INVALID);
	  return sign == 1 ? LONG_MAX : LONG_MIN;
	}
#endif
      return (long int) x;
    }

  return sign * result;
}

libm_alias_float (__lround, lround)
