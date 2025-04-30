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
#include <libm-alias-float.h>


#if LIBM_SVID_COMPAT && SHLIB_COMPAT (libm, GLIBC_2_0, GLIBC_2_27)
/* wrapper logf(x) */
float
__logf_compat (float x)
{
  if (__builtin_expect (islessequal (x, 0.0f), 0) && _LIB_VERSION != _IEEE_)
    {
      if (x == 0.0f)
	{
	  feraiseexcept (FE_DIVBYZERO);
	  return __kernel_standard_f (x, x, 116); /* log(0) */
	}
      else
	{
	  feraiseexcept (FE_INVALID);
	  return __kernel_standard_f (x, x, 117); /* log(x<0) */
	}
    }

  return  __ieee754_logf (x);
}
compat_symbol (libm, __logf_compat, logf, GLIBC_2_0);
#endif
