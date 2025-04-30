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
#include <fenv_private.h>
#include <math-underflow.h>
#include <libm-alias-double.h>

#define IN_SINCOS
#include "s_sin.c"

void
__sincos (double x, double *sinx, double *cosx)
{
  mynumber u;
  int k;

  SET_RESTORE_ROUND_53BIT (FE_TONEAREST);

  u.x = x;
  k = u.i[HIGH_HALF] & 0x7fffffff;

  if (k < 0x400368fd)
    {
      double a, da, y;
      /* |x| < 2^-27 => cos (x) = 1, sin (x) = x.  */
      if (k < 0x3e400000)
	{
	  if (k < 0x3e500000)
	    math_check_force_underflow (x);
	  *sinx = x;
	  *cosx = 1.0;
	  return;
	}
      /* |x| < 0.855469.  */
      else if (k < 0x3feb6000)
	{
	  *sinx = do_sin (x, 0);
	  *cosx = do_cos (x, 0);
	  return;
	}

      /* |x| < 2.426265.  */
      y = hp0 - fabs (x);
      a = y + hp1;
      da = (y - a) + hp1;
      *sinx = copysign (do_cos (a, da), x);
      *cosx = do_sin (a, da);
      return;
    }
  /* |x| < 2^1024.  */
  if (k < 0x7ff00000)
    {
      double a, da, xx;
      unsigned int n;

      /* If |x| < 105414350 use simple range reduction.  */
      n = k < 0x419921FB ? reduce_sincos (x, &a, &da) : __branred (x, &a, &da);
      n = n & 3;

      if (n == 1 || n == 2)
	{
	  a = -a;
	  da = -da;
	}

      if (n & 1)
	{
	  double *temp = cosx;
	  cosx = sinx;
	  sinx = temp;
	}

      *sinx = do_sin (a, da);
      xx = do_cos (a, da);
      *cosx = (n & 2) ? -xx : xx;
      return;
    }

  if (isinf (x))
    __set_errno (EDOM);

  *sinx = *cosx = x / x;
}
libm_alias_double (__sincos, sincos)
