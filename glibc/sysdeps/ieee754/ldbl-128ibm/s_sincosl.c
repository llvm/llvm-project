/* Compute sine and cosine of argument.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Ulrich Drepper <drepper@cygnus.com>, 1997 and
		  Jakub Jelinek <jj@ultra.linux.cz>.

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
#include <math_ldbl_opt.h>

void
__sincosl (long double x, long double *sinx, long double *cosx)
{
  int64_t ix;
  double xhi;

  /* High word of x. */
  xhi = ldbl_high (x);
  EXTRACT_WORDS64 (ix, xhi);

  /* |x| ~< pi/4 */
  ix &= 0x7fffffffffffffffLL;
  if (ix <= 0x3fe921fb54442d10LL)
    __kernel_sincosl (x, 0.0L, sinx, cosx, 0);
  else if (ix >= 0x7ff0000000000000LL)
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
	  __kernel_sincosl (y[0], y[1], sinx, cosx, 1);
	  break;
	case 1:
	  __kernel_sincosl (y[0], y[1], cosx, sinx, 1);
	  *cosx = -*cosx;
	  break;
	case 2:
	  __kernel_sincosl (y[0], y[1], sinx, cosx, 1);
	  *sinx = -*sinx;
	  *cosx = -*cosx;
	  break;
	default:
	  __kernel_sincosl (y[0], y[1], cosx, sinx, 1);
	  *sinx = -*sinx;
	  break;
	}
    }
}
long_double_symbol (libm, __sincosl, sincosl);
