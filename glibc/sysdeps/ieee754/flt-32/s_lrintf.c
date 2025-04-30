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

#include <math-narrow-eval.h>
#include <math_private.h>
#include <libm-alias-float.h>
#include <fix-fp-int-convert-overflow.h>

static const float two23[2] =
{
  8.3886080000e+06, /* 0x4B000000 */
 -8.3886080000e+06, /* 0xCB000000 */
};


long int
__lrintf (float x)
{
  int32_t j0;
  uint32_t i0;
  float w;
  float t;
  long int result;
  int sx;

  GET_FLOAT_WORD (i0, x);

  sx = i0 >> 31;
  j0 = ((i0 >> 23) & 0xff) - 0x7f;
  i0 &= 0x7fffff;
  i0 |= 0x800000;

  if (j0 < (int32_t) (sizeof (long int) * 8) - 1)
    {
      if (j0 >= 23)
	result = (long int) i0 << (j0 - 23);
      else
	{
	  w = math_narrow_eval (two23[sx] + x);
	  t = w - two23[sx];
	  GET_FLOAT_WORD (i0, t);
	  j0 = ((i0 >> 23) & 0xff) - 0x7f;
	  i0 &= 0x7fffff;
	  i0 |= 0x800000;

	  result = (j0 < 0 ? 0 : i0 >> (23 - j0));
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
	  return sx == 0 ? LONG_MAX : LONG_MIN;
	}
#endif
      return (long int) x;
    }

  return sx ? -result : result;
}

libm_alias_float (__lrint, lrint)
