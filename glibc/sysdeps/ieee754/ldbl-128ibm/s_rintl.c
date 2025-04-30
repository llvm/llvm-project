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
#include <fenv.h>
#include <math-barriers.h>
#include <math_private.h>
#include <fenv_private.h>
#include <math_ldbl_opt.h>
#include <float.h>
#include <ieee754.h>

#ifdef USE_AS_NEARBYINTL
# define rintl nearbyintl
# define __rintl __nearbyintl
#endif


long double
__rintl (long double x)
{
  double xh, xl, hi, lo;

  ldbl_unpack (x, &xh, &xl);

  /* Return Inf, Nan, +/-0 unchanged.  */
  if (__builtin_expect (xh != 0.0
			&& __builtin_isless (__builtin_fabs (xh),
					     __builtin_inf ()), 1))
    {
      double orig_xh;
      int save_round = fegetround ();

      /* Long double arithmetic, including the canonicalisation below,
	 only works in round-to-nearest mode.  */
#ifdef USE_AS_NEARBYINTL
      SET_RESTORE_ROUND_NOEX (FE_TONEAREST);
#else
      fesetround (FE_TONEAREST);
#endif

      /* Convert the high double to integer.  */
      orig_xh = xh;
      hi = ldbl_nearbyint (xh);

      /* Subtract integral high part from the value.  If the low double
	 happens to be exactly 0.5 or -0.5, you might think that this
	 subtraction could result in an incorrect conversion.  For
	 instance, subtracting an odd number would cause this function
	 to round in the wrong direction.  However, if we have a
	 canonical long double with the low double 0.5 or -0.5, then the
	 high double must be even.  */
      xh -= hi;
      ldbl_canonicalize (&xh, &xl);

      /* Now convert the low double, adjusted for any remainder from the
	 high double.  */
      lo = ldbl_nearbyint (xh);

      xh -= lo;
      ldbl_canonicalize (&xh, &xl);

      switch (save_round)
	{
	case FE_TONEAREST:
	  if (xl > 0.0 && xh == 0.5)
	    lo += 1.0;
	  else if (xl < 0.0 && -xh == 0.5)
	    lo -= 1.0;
	  break;

	case FE_TOWARDZERO:
	  if (orig_xh < 0.0)
	    goto do_up;
	  /* Fall thru */

	case FE_DOWNWARD:
	  if (xh < 0.0 || (xh == 0.0 && xl < 0.0))
	    lo -= 1.0;
	  break;

	case FE_UPWARD:
	do_up:
	  if (xh > 0.0 || (xh == 0.0 && xl > 0.0))
	    lo += 1.0;
	  break;
	}

      /* Ensure the final value is canonical.  In certain cases,
         rounding causes hi,lo calculated so far to be non-canonical.  */
      xh = hi;
      xl = lo;
      ldbl_canonicalize (&xh, &xl);

      /* Ensure we return -0 rather than +0 when appropriate.  */
      if (orig_xh < 0.0)
	xh = -__builtin_fabs (xh);

#ifdef USE_AS_NEARBYINTL
      math_force_eval (xh);
      math_force_eval (xl);
#else
      fesetround (save_round);
#endif
    }
  else
    /* Quiet signaling NaN arguments.  */
    xh += xh;

  return ldbl_pack (xh, xl);
}

long_double_symbol (libm, __rintl, rintl);
