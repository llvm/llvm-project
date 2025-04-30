/* Copyright (C) 2005-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Richard Henderson <rth@redhat.com>, 2005.

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

#include <stdbool.h>
#include <math.h>
#include <complex.h>

attribute_hidden
long double _Complex
__multc3 (long double a, long double b, long double c, long double d)
{
  long double ac, bd, ad, bc, x, y;

  ac = a * c;
  bd = b * d;
  ad = a * d;
  bc = b * c;

  x = ac - bd;
  y = ad + bc;

  if (isnan (x) && isnan (y))
    {
      /* Recover infinities that computed as NaN + iNaN.  */
      bool recalc = 0;
      if (isinf (a) || isinf (b))
	{
	  /* z is infinite.  "Box" the infinity and change NaNs in
	     the other factor to 0.  */
	  a = copysignl (isinf (a) ? 1 : 0, a);
	  b = copysignl (isinf (b) ? 1 : 0, b);
	  if (isnan (c)) c = copysignl (0, c);
	  if (isnan (d)) d = copysignl (0, d);
	  recalc = 1;
	}
     if (isinf (c) || isinf (d))
	{
	  /* w is infinite.  "Box" the infinity and change NaNs in
	     the other factor to 0.  */
	  c = copysignl (isinf (c) ? 1 : 0, c);
	  d = copysignl (isinf (d) ? 1 : 0, d);
	  if (isnan (a)) a = copysignl (0, a);
	  if (isnan (b)) b = copysignl (0, b);
	  recalc = 1;
	}
     if (!recalc
	  && (isinf (ac) || isinf (bd)
	      || isinf (ad) || isinf (bc)))
	{
	  /* Recover infinities from overflow by changing NaNs to 0.  */
	  if (isnan (a)) a = copysignl (0, a);
	  if (isnan (b)) b = copysignl (0, b);
	  if (isnan (c)) c = copysignl (0, c);
	  if (isnan (d)) d = copysignl (0, d);
	  recalc = 1;
	}
      if (recalc)
	{
	  x = INFINITY * (a * c - b * d);
	  y = INFINITY * (a * d + b * c);
	}
    }

  return x + I * y;
}
