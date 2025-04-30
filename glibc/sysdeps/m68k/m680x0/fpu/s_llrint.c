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
#include <libm-alias-double.h>
#include "mathimpl.h"

long long int
__llrint (double x)
{
  int32_t e;
  uint32_t h, l, s;
  long long int result;

  x = __m81_u(__rint) (x);

  /* We could use __fixxfdi from libgcc, but here we can take advantage of
     the known floating point format.  */
  EXTRACT_WORDS (h, l, x);

  e = ((h >> 20) & 0x7ff) - 0x3ff;
  if (e < 0)
    return 0;
  s = h;
  h &= 0xfffff;
  h |= 0x100000;

  if (e < 63)
    {
      if (e > 52)
	{
	  h <<= e - 52;
	  h |= l >> (84 - e);
	  l <<= e - 52;
	  result = ((long long int) h << 32) | l;
	}
      else if (e > 20)
	{
	  l >>= 52 - e;
	  l |= h << (e - 20);
	  h >>= 52 - e;
	  result = ((long long int) h << 32) | l;
	}
      else
	result = h >> (20 - e);
      if (s & 0x80000000)
	result = -result;
    }
  else
    /* The number is too large or not finite.  The standard leaves it
       undefined what to return when the number is too large to fit in a
       `long long int'.  */
    result = -1LL;

  return result;
}

libm_alias_double (__llrint, llrint)
