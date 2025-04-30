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
#include <libm-alias-double.h>


#if LIBM_SVID_COMPAT && (SHLIB_COMPAT (libm, GLIBC_2_0, GLIBC_2_29) \
			 || defined NO_LONG_DOUBLE \
			 || defined LONG_DOUBLE_COMPAT)
/* wrapper pow */
double
__pow_compat (double x, double y)
{
  double z = __ieee754_pow (x, y);
  if (__glibc_unlikely (!isfinite (z)))
    {
      if (_LIB_VERSION != _IEEE_)
	{
	  if (isfinite (x) && isfinite (y))
	    {
	      if (isnan (z))
		/* pow neg**non-int */
		return __kernel_standard (x, y, 24);
	      else if (x == 0.0 && y < 0.0)
		{
		  if (signbit (x) && signbit (z))
		    /* pow(-0.0,negative) */
		    return __kernel_standard (x, y, 23);
		  else
		    /* pow(+0.0,negative) */
		    return __kernel_standard (x, y, 43);
		}
	      else
		/* pow overflow */
		return __kernel_standard (x, y, 21);
	    }
	}
    }
  else if (__builtin_expect (z == 0.0, 0)
	   && isfinite (x) && x != 0 && isfinite (y)
	   && _LIB_VERSION != _IEEE_)
    /* pow underflow */
    return __kernel_standard (x, y, 22);

  return z;
}
# if SHLIB_COMPAT (libm, GLIBC_2_0, GLIBC_2_29)
compat_symbol (libm, __pow_compat, pow, GLIBC_2_0);
# endif
# ifdef NO_LONG_DOUBLE
weak_alias (__pow_compat, powl)
# endif
# ifdef LONG_DOUBLE_COMPAT
/* Work around gas bug "multiple versions for symbol".  */
weak_alias (__pow_compat, __pow_compat_alias)

LONG_DOUBLE_COMPAT_CHOOSE_libm_powl (
  compat_symbol (libm, __pow_compat_alias, powl, FIRST_VERSION_libm_powl), );
# endif
#endif
