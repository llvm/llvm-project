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

#include <fenv.h>
#include <math.h>
#include <math_private.h>
#include <math-svid-compat.h>
#include <libm-alias-double.h>


#if LIBM_SVID_COMPAT && (SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_29) \
			 || defined NO_LONG_DOUBLE \
			 || defined LONG_DOUBLE_COMPAT)
/* wrapper log2(x) */
double
__log2_compat (double x)
{
  if (__builtin_expect (islessequal (x, 0.0), 0) && _LIB_VERSION != _IEEE_)
    {
      if (x == 0.0)
	{
	  feraiseexcept (FE_DIVBYZERO);
	  return __kernel_standard (x, x, 48); /* log2(0) */
	}
      else
	{
	  feraiseexcept (FE_INVALID);
	  return __kernel_standard (x, x, 49); /* log2(x<0) */
	}
    }

  return  __ieee754_log2 (x);
}
# if SHLIB_COMPAT (libm, GLIBC_2_1, GLIBC_2_29)
compat_symbol (libm, __log2_compat, log2, GLIBC_2_1);
# endif
# ifdef NO_LONG_DOUBLE
weak_alias (__log2_compat, log2l)
# endif
# ifdef LONG_DOUBLE_COMPAT
/* Work around gas bug "multiple versions for symbol".  */
weak_alias (__log2_compat, __log2_compat_alias)

LONG_DOUBLE_COMPAT_CHOOSE_libm_log2l (
  compat_symbol (libm, __log2_compat_alias, log2l, FIRST_VERSION_libm_log2l), );
# endif
#endif
