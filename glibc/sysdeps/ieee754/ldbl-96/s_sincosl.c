/* Compute sine and cosine of argument.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997.

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
#include <libm-alias-ldouble.h>


void
__sincosl (long double x, long double *sinx, long double *cosx)
{
  int32_t se, i0, i1 __attribute__ ((unused));

  /* High word of x. */
  GET_LDOUBLE_WORDS (se, i0, i1, x);

  /* |x| ~< pi/4 */
  se &= 0x7fff;
  if (se < 0x3ffe || (se == 0x3ffe && i0 <= 0xc90fdaa2))
    {
      *sinx = __kernel_sinl (x, 0.0, 0);
      *cosx = __kernel_cosl (x, 0.0);
    }
  else if (se == 0x7fff)
    {
      /* sin(Inf or NaN) is NaN */
      *sinx = *cosx = x - x;
      if (isinf (x))
	__set_errno (EDOM);
    }
  else
    {
      /* Argument reduction needed.  */
      long double y[2];
      int n;

      n = __ieee754_rem_pio2l (x, y);
      switch (n & 3)
	{
	case 0:
	  *sinx = __kernel_sinl (y[0], y[1], 1);
	  *cosx = __kernel_cosl (y[0], y[1]);
	  break;
	case 1:
	  *sinx = __kernel_cosl (y[0], y[1]);
	  *cosx = -__kernel_sinl (y[0], y[1], 1);
	  break;
	case 2:
	  *sinx = -__kernel_sinl (y[0], y[1], 1);
	  *cosx = -__kernel_cosl (y[0], y[1]);
	  break;
	default:
	  *sinx = -__kernel_cosl (y[0], y[1]);
	  *cosx = __kernel_sinl (y[0], y[1], 1);
	  break;
	}
    }
}
libm_alias_ldouble (__sincos, sincos)
