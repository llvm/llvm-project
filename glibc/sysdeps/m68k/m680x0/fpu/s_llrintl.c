/* Round argument to nearest integral value according to current rounding
   direction.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@issan.informatik.uni-dortmund.de>

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <math.h>
#include <math_private.h>
#include <libm-alias-ldouble.h>
#include "mathimpl.h"

long long int
__llrintl (long double x)
{
  int32_t e, s;
  uint32_t h, l;
  long long int result;

  x = __m81_u(__rintl) (x);

  GET_LDOUBLE_WORDS (e, h, l, x);

  s = e;
  e = (e & 0x7fff) - 0x3fff;
  if (e < 0)
    return 0;

  if (e < 63)
    {
      if (e > 31)
	{
	  l >>= 63 - e;
	  l |= h << (e - 31);
	  h >>= 63 - e;
	  result = ((long long int) h << 32) | l;
	}
      else
	result = h >> (31 - e);
      if (s & 0x8000)
	result = -result;
    }
  else
    /* The number is too large or not finite.  The standard leaves it
       undefined what to return when the number is too large to fit in a
       `long long int'.  */
    result = -1LL;

  return result;
}

libm_alias_ldouble (__llrint, llrint)
