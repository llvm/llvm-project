/* Implement __kernel_standard_l.
   Copyright (C) 2012-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

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
   <https://www.gnu.org/licenses/>.

   Parts based on k_standard.c from fdlibm: */

/* @(#)k_standard.c 5.1 93/09/24 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

#include <math.h>
#include <math-barriers.h>
#include <math-svid-compat.h>
#include <fenv.h>
#include <float.h>
#include <errno.h>


#if LIBM_SVID_COMPAT

static double zero = 0.0;

/* Handle errors for a libm function as specified by TYPE (see
   comments in k_standard.c for details), with arguments X and Y,
   returning the appropriate return value for that function.  */

long double
__kernel_standard_l (long double x, long double y, int type)
{
  double dx, dy;
  struct exception exc;
  fenv_t env;

  feholdexcept (&env);
  dx = x;
  dy = y;
  math_force_eval (dx);
  math_force_eval (dy);
  fesetenv (&env);

  switch (type)
    {
    case 221:
      /* powl (x, y) overflow.  */
      exc.arg1 = dx;
      exc.arg2 = dy;
      exc.type = OVERFLOW;
      exc.name = (char *) "powl";
      if (_LIB_VERSION == _SVID_)
	{
	  exc.retval = HUGE;
	  y *= 0.5;
	  if (x < zero && rintl (y) != y)
	    exc.retval = -HUGE;
	}
      else
	{
	  exc.retval = HUGE_VAL;
	  y *= 0.5;
	  if (x < zero && rintl (y) != y)
	    exc.retval = -HUGE_VAL;
	}
      if (_LIB_VERSION == _POSIX_)
	__set_errno (ERANGE);
      else if (!matherr (&exc))
	__set_errno (ERANGE);
      return exc.retval;

    case 222:
      /* powl (x, y) underflow.  */
      exc.arg1 = dx;
      exc.arg2 = dy;
      exc.type = UNDERFLOW;
      exc.name = (char *) "powl";
      exc.retval = zero;
      y *= 0.5;
      if (x < zero && rintl (y) != y)
	exc.retval = -zero;
      if (_LIB_VERSION == _POSIX_)
	__set_errno (ERANGE);
      else if (!matherr (&exc))
	__set_errno (ERANGE);
      return exc.retval;

    default:
      return __kernel_standard (dx, dy, type);
    }
}
#endif
