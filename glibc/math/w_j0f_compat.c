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


#if LIBM_SVID_COMPAT
/* wrapper j0f */
float
__j0f (float x)
{
  if (__builtin_expect (isgreater (fabsf (x), (float) X_TLOSS), 0)
      && _LIB_VERSION != _IEEE_ && _LIB_VERSION != _POSIX_)
    /* j0(|x|>X_TLOSS) */
    return __kernel_standard_f (x, x, 134);

  return __ieee754_j0f (x);
}
libm_alias_float (__j0, j0)


/* wrapper y0f */
float
__y0f (float x)
{
  if (__builtin_expect (islessequal (x, 0.0f)
                        || isgreater (x, (float) X_TLOSS), 0)
      && _LIB_VERSION != _IEEE_)
    {
      if (x < 0.0f)
	{
	  /* d = zero/(x-x) */
	  feraiseexcept (FE_INVALID);
	  return __kernel_standard_f (x, x, 109);
	}
      else if (x == 0.0f)
	{
	  /* d = -one/(x-x) */
	  feraiseexcept (FE_DIVBYZERO);
	  return __kernel_standard_f (x, x, 108);
	}
      else if (_LIB_VERSION != _POSIX_)
	/* y0(x>X_TLOSS) */
	return __kernel_standard_f (x, x, 135);
    }

  return __ieee754_y0f (x);
}
libm_alias_float (__y0, y0)
#endif
