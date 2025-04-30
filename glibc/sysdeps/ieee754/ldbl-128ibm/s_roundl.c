/* Round to int long double floating-point values.
   IBM extended format long double version.
   Copyright (C) 2006-2021 Free Software Foundation, Inc.
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

/* This has been coded in assembler because GCC makes such a mess of it
   when it's coded in C.  */

#define NO_MATH_REDIRECT
#include <math.h>
#include <math_private.h>
#include <math_ldbl_opt.h>
#include <float.h>
#include <ieee754.h>

double round (double) asm ("__round");


long double
__roundl (long double x)
{
  double xh, xl, hi, lo;

  ldbl_unpack (x, &xh, &xl);

  /* Return Inf, Nan, +/-0 unchanged.  */
  if (__builtin_expect (xh != 0.0
			&& __builtin_isless (__builtin_fabs (xh),
					     __builtin_inf ()), 1))
    {
      hi = round (xh);
      if (hi != xh)
	{
	  /* The high part is not an integer; the low part only
	     affects the result if the high part is exactly half way
	     between two integers and the low part is nonzero with the
	     opposite sign.  */
	  if (fabs (hi - xh) == 0.5)
	    {
	      if (xh > 0 && xl < 0)
		xh = hi - 1;
	      else if (xh < 0 && xl > 0)
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
	  /* The high part is a nonzero integer.  */
	  lo = round (xl);
	  if (fabs (lo - xl) == 0.5)
	    {
	      if (xh > 0 && xl < 0)
		xl = lo + 1;
	      else if (xh < 0 && lo > 0)
		xl = lo - 1;
	      else
		xl = lo;
	    }
	  else
	    xl = lo;
	  xh = hi;
	  ldbl_canonicalize_int (&xh, &xl);
	}
    }
  else
    /* Quiet signaling NaN arguments.  */
    xh += xh;

  return ldbl_pack (xh, xl);
}

long_double_symbol (libm, __roundl, roundl);
