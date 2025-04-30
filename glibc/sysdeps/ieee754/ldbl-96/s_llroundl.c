/* Round long double value to long long int.
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
#include <libm-alias-ldouble.h>


long long int
__llroundl (long double x)
{
  int32_t j0;
  uint32_t se, i1, i0;
  long long int result;
  int sign;

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  j0 = (se & 0x7fff) - 0x3fff;
  sign = (se & 0x8000) != 0 ? -1 : 1;

  if (j0 < 31)
    {
      if (j0 < 0)
	return j0 < -1 ? 0 : sign;
      else
	{
	  uint32_t j = i0 + (0x40000000 >> j0);
	  if (j < i0)
	    {
	      j >>= 1;
	      j |= 0x80000000;
	      ++j0;
	    }

	  result = j >> (31 - j0);
	}
    }
  else if (j0 < (int32_t) (8 * sizeof (long long int)) - 1)
    {
      if (j0 >= 63)
	result = (((long long int) i0 << 32) | i1) << (j0 - 63);
      else
	{
	  uint32_t j = i1 + (0x80000000 >> (j0 - 31));

	  result = (long long int) i0;
	  if (j < i1)
	    ++result;

	  if (j0 > 31)
	    {
	      result = (result << (j0 - 31)) | (j >> (63 - j0));
#ifdef FE_INVALID
	      if (sign == 1 && result == LLONG_MIN)
		/* Rounding brought the value out of range.  */
		feraiseexcept (FE_INVALID);
#endif
	    }
	}
    }
  else
    {
      /* The number is too large.  It is left implementation defined
	 what happens.  */
      return (long long int) x;
    }

  return sign * result;
}

libm_alias_ldouble (__llround, llround)
