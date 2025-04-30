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
#include <libm-alias-float.h>


#if LIBM_SVID_COMPAT && SHLIB_COMPAT (libm, GLIBC_2_0, GLIBC_2_27)
/* wrapper powf */
float
__powf_compat (float x, float y)
{
  float z = __ieee754_powf (x, y);
  if (__glibc_unlikely (!isfinite (z)))
    {
      if (_LIB_VERSION != _IEEE_)
	{
	  if (isfinite (x) && isfinite (y))
	    {
	      if (isnan (z))
		/* pow neg**non-int */
		return __kernel_standard_f (x, y, 124);
	      else if (x == 0.0f && y < 0.0f)
		{
		  if (signbit (x) && signbit (z))
		    /* pow(-0.0,negative) */
		    return __kernel_standard_f (x, y, 123);
		  else
		    /* pow(+0.0,negative) */
		    return __kernel_standard_f (x, y, 143);
		}
	      else
		/* pow overflow */
		return __kernel_standard_f (x, y, 121);
	    }
	}
    }
  else if (__builtin_expect (z == 0.0f, 0)
	   && isfinite (x) && x != 0 && isfinite (y)
	   && _LIB_VERSION != _IEEE_)
    /* pow underflow */
    return __kernel_standard_f (x, y, 122);

  return z;
}
compat_symbol (libm, __powf_compat, powf, GLIBC_2_0);
#endif
