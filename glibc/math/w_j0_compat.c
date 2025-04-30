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
/* wrapper j0 */
double
__j0 (double x)
{
  if (__builtin_expect (isgreater (fabs (x), X_TLOSS), 0)
      && _LIB_VERSION != _IEEE_ && _LIB_VERSION != _POSIX_)
    /* j0(|x|>X_TLOSS) */
    return __kernel_standard (x, x, 34);

  return __ieee754_j0 (x);
}
libm_alias_double (__j0, j0)


/* wrapper y0 */
double
__y0 (double x)
{
  if (__builtin_expect (islessequal (x, 0.0) || isgreater (x, X_TLOSS), 0)
      && _LIB_VERSION != _IEEE_)
    {
      if (x < 0.0)
	{
	  /* d = zero/(x-x) */
	  __feraiseexcept (FE_INVALID);
	  return __kernel_standard (x, x, 9);
	}
      else if (x == 0.0)
	{
	  /* d = -one/(x-x) */
	  __feraiseexcept (FE_DIVBYZERO);
	  return __kernel_standard (x, x, 8);
	}
      else if (_LIB_VERSION != _POSIX_)
	/* y0(x>X_TLOSS) */
	return __kernel_standard (x, x, 35);
    }

  return __ieee754_y0 (x);
}
libm_alias_double (__y0, y0)
#endif
