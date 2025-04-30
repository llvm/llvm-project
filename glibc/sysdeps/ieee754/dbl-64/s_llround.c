/* Round double value to long long int.
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

#define lround __hidden_lround
#define __lround __hidden___lround

#include <fenv.h>
#include <limits.h>
#include <math.h>
#include <sysdep.h>

#include <math_private.h>
#include <libm-alias-double.h>
#include <fix-fp-int-convert-overflow.h>

long long int
__llround (double x)
{
  int32_t j0;
  int64_t i0;
  long long int result;
  int sign;

  EXTRACT_WORDS64 (i0, x);
  j0 = ((i0 >> 52) & 0x7ff) - 0x3ff;
  sign = i0 < 0 ? -1 : 1;
  i0 &= UINT64_C(0xfffffffffffff);
  i0 |= UINT64_C(0x10000000000000);

  if (j0 < (int32_t) (8 * sizeof (long long int)) - 1)
    {
      if (j0 < 0)
	return j0 < -1 ? 0 : sign;
      else if (j0 >= 52)
	result = i0 << (j0 - 52);
      else
	{
	  i0 += UINT64_C(0x8000000000000) >> j0;

	  result = i0 >> (52 - j0);
	}
    }
  else
    {
#ifdef FE_INVALID
      /* The number is too large.  Unless it rounds to LLONG_MIN,
	 FE_INVALID must be raised and the return value is
	 unspecified.  */
      if (FIX_DBL_LLONG_CONVERT_OVERFLOW && x != (double) LLONG_MIN)
	{
	  feraiseexcept (FE_INVALID);
	  return sign == 1 ? LLONG_MAX : LLONG_MIN;
	}
#endif
      return (long long int) x;
    }

  return sign * result;
}

libm_alias_double (__llround, llround)

/* long has the same width as long long on LP64 machines, so use an alias.  */
#undef lround
#undef __lround
#ifdef _LP64
strong_alias (__llround, __lround)
libm_alias_double (__lround, lround)
#endif
