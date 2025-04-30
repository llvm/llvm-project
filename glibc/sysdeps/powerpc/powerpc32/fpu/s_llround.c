/* Round double value to long long int.
   Copyright (C) 1997-2021 Free Software Foundation, Inc.
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

#include <limits.h>
#include <math.h>
#include <math_private.h>
#include <stdint.h>
#include <libm-alias-double.h>
#include <math-barriers.h>

/* Round to the nearest integer, with values exactly on a 0.5 boundary
   rounded away from zero, regardless of the current rounding mode.
   If (long long)x, when x is out of range of a long long, clips at
   LLONG_MAX or LLONG_MIN, then this implementation also clips.  */

long long int
__llround (double x)
{
#ifdef _ARCH_PWR5X
  x = round (x);
  /* The barrier prevents compiler from optimizing it to llround when
     compiled with -fno-math-errno */
  math_opt_barrier (x);
  return x;
#else
  long long xr;
  if (HAVE_PPC_FCTIDZ)
    {
       /* IEEE 1003.1 lround function.  IEEE specifies "round to the nearest
	  integer value, rounding halfway cases away from zero, regardless of
	  the current rounding mode."  However PowerPC Architecture defines
	  "round to Nearest" as "Choose the best approximation. In case of a
	  tie, choose the one that is even (least significant bit o).".
	  So we can't use the PowerPC "round to Nearest" mode. Instead we set
	  "round toward Zero" mode and round by adding +-0.5 before rounding
	  to the integer value.

	  It is necessary to detect when x is (+-)0x1.fffffffffffffp-2
	  because adding +-0.5 in this case will cause an erroneous shift,
	  carry and round.  We simply return 0 if 0.5 > x > -0.5.  Likewise
	  if x is and odd number between +-(2^52 and 2^53-1) a shift and
	  carry will erroneously round if biased with +-0.5.  Therefore if x
	  is greater/less than +-2^52 we don't need to bias the number with
	  +-0.5.  */
      double ax = fabs (x);

      if (ax < 0.5)
	return 0;

      if (ax < 0x1p+52)
	{
	  /* Test whether an integer to avoid spurious "inexact".  */
	  double t = ax + 0x1p+52;
	  t = t - 0x1p+52;
	  if (ax != t)
	    {
	      ax = ax + 0.5;
	      if (x < 0.0)
		ax = -fabs (ax);
	      x = ax;
	    }
        }

      return x;
    }
  else
    {
      /* Avoid incorrect exceptions from libgcc conversions (as of GCC
	 5): <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59412>.  */
      if (fabs (x) < 0x1p31)
	xr = (long long int) (long int) x;
      else
	{
	  uint64_t i0;
	  EXTRACT_WORDS64 (i0, x);
	  int exponent = ((i0 >> 52) & 0x7ff) - 0x3ff;
	  if (exponent < 63)
	    {
	      unsigned long long int mant
		= (i0 & ((1ULL << 52) - 1)) | (1ULL << 52);
	      if (exponent < 52)
		/* llround is not required to raise "inexact".  */
		mant >>= 52 - exponent;
	      else
		mant <<= exponent - 52;
	      xr = (long long int) ((i0 & (1ULL << 63)) != 0 ? -mant : mant);
	    }
	  else if (x == (double) LLONG_MIN)
	    xr = LLONG_MIN;
	  else
	    xr = (long long int) (long int) x << 32;
	}
    }
  /* Avoid spurious "inexact" converting LLONG_MAX to double, and from
     subtraction when the result is out of range, by returning early
     for arguments large enough that no rounding is needed.  */
  if (!(fabs (x) < 0x1p52))
    return xr;
  double xrf = (double) xr;

  if (x >= 0.0)
    {
      if (x - xrf >= 0.5)
	xr += (long long) ((unsigned long long) xr + 1) > 0;
    }
  else
    {
      if (xrf - x >= 0.5)
	xr -= (long long) ((unsigned long long) xr - 1) < 0;
    }
  return xr;
#endif
}
#ifndef __llround
libm_alias_double (__llround, llround)
#endif
