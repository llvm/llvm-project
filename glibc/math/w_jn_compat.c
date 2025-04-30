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


#if LIBM_SVID_COMPAT
/* wrapper jn */
double
__jn (int n, double x)
{
  if (__builtin_expect (isgreater (fabs (x), X_TLOSS), 0)
      && _LIB_VERSION != _IEEE_ && _LIB_VERSION != _POSIX_)
    /* jn(n,|x|>X_TLOSS) */
    return __kernel_standard (n, x, 38);

  return __ieee754_jn (n, x);
}
libm_alias_double (__jn, jn)


/* wrapper yn */
double
__yn (int n, double x)
{
  if (__builtin_expect (islessequal (x, 0.0) || isgreater (x, X_TLOSS), 0)
      && _LIB_VERSION != _IEEE_)
    {
      if (x < 0.0)
	{
	  /* d = zero/(x-x) */
	  __feraiseexcept (FE_INVALID);
	  return __kernel_standard (n, x, 13);
	}
      else if (x == 0.0)
	{
	  /* d = -one/(x-x) */
	  __feraiseexcept (FE_DIVBYZERO);
	  return __kernel_standard (n, x, 12);
	}
      else if (_LIB_VERSION != _POSIX_)
	/* yn(n,x>X_TLOSS) */
	return __kernel_standard (n, x, 39);
    }

  return __ieee754_yn (n, x);
}
libm_alias_double (__yn, yn)
#endif
