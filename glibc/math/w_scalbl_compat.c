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

#include <errno.h>
#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>


#if LIBM_SVID_COMPAT
static long double
__attribute__ ((noinline))
sysv_scalbl (long double x, long double fn)
{
  long double z = __ieee754_scalbl (x, fn);

  if (__glibc_unlikely (isinf (z)))
    {
      if (isfinite (x))
	return __kernel_standard_l (x, fn, 232); /* scalb overflow */
      else
	__set_errno (ERANGE);
    }
  else if (__builtin_expect (z == 0.0L, 0) && z != x)
    return __kernel_standard_l (x, fn, 233); /* scalb underflow */

  return z;
}


/* Wrapper scalbl */
long double
__scalbl (long double x, long double fn)
{
  if (__glibc_unlikely (_LIB_VERSION == _SVID_))
    return sysv_scalbl (x, fn);
  else
    {
      long double z = __ieee754_scalbl (x, fn);

      if (__glibc_unlikely (!isfinite (z) || z == 0.0L))
	{
	  if (isnan (z))
	    {
	      if (!isnan (x) && !isnan (fn))
		__set_errno (EDOM);
	    }
	  else if (isinf (z))
	    {
	      if (!isinf (x) && !isinf (fn))
		__set_errno (ERANGE);
	    }
	  else
	    {
	      /* z == 0.  */
	      if (x != 0.0L && !isinf (fn))
		__set_errno (ERANGE);
	    }
	}
      return z;
    }
}
weak_alias (__scalbl, scalbl)
#endif
