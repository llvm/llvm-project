/* Compute cubic root of double value.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Dirk Alboth <dirka@uni-paderborn.de> and
   Ulrich Drepper <drepper@cygnus.com>, 1997.

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

#include <math.h>
#include <libm-alias-double.h>


#define CBRT2 1.2599210498948731648		/* 2^(1/3) */
#define SQR_CBRT2 1.5874010519681994748		/* 2^(2/3) */

static const double factor[5] =
{
  1.0 / SQR_CBRT2,
  1.0 / CBRT2,
  1.0,
  CBRT2,
  SQR_CBRT2
};


double
__cbrt (double x)
{
  double xm, ym, u, t2;
  int xe;

  /* Reduce X.  XM now is an range 1.0 to 0.5.  */
  xm = __frexp (fabs (x), &xe);

  /* If X is not finite or is null return it (with raising exceptions
     if necessary.
     Note: *Our* version of `frexp' sets XE to zero if the argument is
     Inf or NaN.  This is not portable but faster.  */
  if (xe == 0 && fpclassify (x) <= FP_ZERO)
    return x + x;

  u = (0.354895765043919860
       + ((1.50819193781584896
	   + ((-2.11499494167371287
	       + ((2.44693122563534430
		   + ((-1.83469277483613086
		       + (0.784932344976639262 - 0.145263899385486377 * xm)
	                  * xm)
		      * xm))
		  * xm))
	      * xm))
	  * xm));

  t2 = u * u * u;

  ym = u * (t2 + 2.0 * xm) / (2.0 * t2 + xm) * factor[2 + xe % 3];

  return __ldexp (x > 0.0 ? ym : -ym, xe / 3);
}
libm_alias_double (__cbrt, cbrt)
