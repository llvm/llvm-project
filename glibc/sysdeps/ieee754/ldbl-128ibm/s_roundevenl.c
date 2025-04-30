/* Round to nearest integer value, rounding halfway cases to even.
   ldbl-128ibm version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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
   <https://www.gnu.org/licenses/>.  */

#define NO_MATH_REDIRECT
#include <math.h>
#include <math_private.h>

long double
__roundevenl (long double x)
{
  double xh, xl, hi;

  ldbl_unpack (x, &xh, &xl);

  if (xh != 0 && isfinite (xh))
    {
      hi = __roundeven (xh);
      if (hi != xh)
	{
	  /* The high part is not an integer; the low part only
	     affects the result if the high part is exactly half way
	     between two integers and the low part is nonzero in the
	     opposite direction to the rounding of the high part.  */
	  double diff = hi - xh;
	  if (fabs (diff) == 0.5)
	    {
	      if (xl < 0 && diff > 0)
		xh = hi - 1;
	      else if (xl > 0 && diff < 0)
		xh = hi + 1;
	      else
		xh = hi;
	    }
	  else
	    xh = hi;
	  xl = 0;
	}
      else
	{
	  /* The high part is a nonzero integer.  Rounding the low
	     part to nearest, ties round to even, is always correct,
	     as a high part that is an odd integer together with a low
	     part with magnitude 0.5 is not a valid long double.  */
	  xl = __roundeven (xl);
	  xh = hi;
	  ldbl_canonicalize_int (&xh, &xl);
	}
    }
  else
    /* Quiet signaling NaN arguments.  */
    xh += xh;

  return ldbl_pack (xh, xl);
}
weak_alias (__roundevenl, roundevenl)
