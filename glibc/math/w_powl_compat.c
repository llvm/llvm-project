/* Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@gmail.com>, 2011.

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
#include <math-svid-compat.h>
#include <libm-alias-ldouble.h>


#if LIBM_SVID_COMPAT
/* wrapper powl */
long double
__powl (long double x, long double y)
{
  long double z = __ieee754_powl (x, y);
  if (__glibc_unlikely (!isfinite (z)))
    {
      if (_LIB_VERSION != _IEEE_)
	{
	  if (isfinite (x) && isfinite (y))
	    {
	      if (isnan (z))
		/* pow neg**non-int */
		return __kernel_standard_l (x, y, 224);
	      else if (x == 0.0L && y < 0.0L)
		{
		  if (signbit (x) && signbit (z))
		    /* pow(-0.0,negative) */
		    return __kernel_standard_l (x, y, 223);
		  else
		    /* pow(+0.0,negative) */
		    return __kernel_standard_l (x, y, 243);
		}
	      else
		/* pow overflow */
		return __kernel_standard_l (x, y, 221);
	    }
	}
    }
  else if (__builtin_expect (z == 0.0L, 0)
	   && isfinite (x) && x != 0 && isfinite (y)
	   && _LIB_VERSION != _IEEE_)
    /* pow underflow */
    return __kernel_standard_l (x, y, 222);

  return z;
}
libm_alias_ldouble (__pow, pow)
#endif
