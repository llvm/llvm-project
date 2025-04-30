/* Round to long long int long double floating-point values.
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

#include <math.h>
#include <fenv.h>
#include <math_private.h>
#include <math_ldbl_opt.h>
#include <float.h>
#include <ieee754.h>

long long
__llroundl (long double x)
{
  double xh, xl;
  long long res, hi, lo;

  ldbl_unpack (x, &xh, &xl);

  /* Limit the range of values handled by the conversion to long long.
     We do this because we aren't sure whether that conversion properly
     raises FE_INVALID.  */
  if (__builtin_expect
      ((__builtin_fabs (xh) <= -(double) (-__LONG_LONG_MAX__ - 1)), 1)
#if !defined (FE_INVALID)
      || 1
#endif
    )
    {
      if (__glibc_unlikely ((xh == -(double) (-__LONG_LONG_MAX__ - 1))))
	{
	  /* When XH is 9223372036854775808.0, converting to long long will
	     overflow, resulting in an invalid operation.  However, XL might
	     be negative and of sufficient magnitude that the overall long
	     double is in fact in range.  Avoid raising an exception.  In any
	     case we need to convert this value specially, because
	     the converted value is not exactly represented as a double
	     thus subtracting HI from XH suffers rounding error.  */
	  hi = __LONG_LONG_MAX__;
	  xh = 1.0;
	}
      else
	{
	  hi = (long long) xh;
	  xh -= hi;
	}
      ldbl_canonicalize (&xh, &xl);

      lo = (long long) xh;

      /* Peg at max/min values, assuming that the above conversions do so.
         Strictly speaking, we can return anything for values that overflow,
         but this is more useful.  */
      res = hi + lo;

      /* This is just sign(hi) == sign(lo) && sign(res) != sign(hi).  */
      if (__glibc_unlikely (((~(hi ^ lo) & (res ^ hi)) < 0)))
	goto overflow;

      xh -= lo;
      ldbl_canonicalize (&xh, &xl);

      hi = res;
      if (xh > 0.5)
	{
	  res += 1;
	}
      else if (xh == 0.5)
	{
	  if (xl > 0.0 || (xl == 0.0 && res >= 0))
	    res += 1;
	}
      else if (-xh > 0.5)
	{
	  res -= 1;
	}
      else if (-xh == 0.5)
	{
	  if (xl < 0.0 || (xl == 0.0 && res <= 0))
	    res -= 1;
	}

      if (__glibc_unlikely (((~(hi ^ (res - hi)) & (res ^ hi)) < 0)))
	goto overflow;

      return res;
    }
  else
    {
      if (xh > 0.0)
	hi = __LONG_LONG_MAX__;
      else if (xh < 0.0)
	hi = -__LONG_LONG_MAX__ - 1;
      else
	/* Nan */
	hi = 0;
    }

overflow:
#ifdef FE_INVALID
  feraiseexcept (FE_INVALID);
#endif
  return hi;
}

long_double_symbol (libm, __llroundl, llroundl);
