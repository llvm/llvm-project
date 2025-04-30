/* Pythagorean addition using doubles
   Copyright (C) 2011-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library
   Contributed by Adhemerval Zanella <azanella@br.ibm.com>, 2011

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Library General Public License as
   published by the Free Software Foundation; either version 2 of the
   License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Library General Public License for more details.

   You should have received a copy of the GNU Library General Public
   License along with the GNU C Library; see the file COPYING.LIB.  If
   not, see <https://www.gnu.org/licenses/>.  */

#include <math.h>
#include <math_private.h>
#include <math-underflow.h>
#include <stdint.h>
#include <libm-alias-finite.h>

/* __ieee754_hypot(x,y)
 *
 * This a FP only version without any FP->INT conversion.
 * It is similar to default C version, making appropriates
 * overflow and underflows checks as well scaling when it
 * is needed.
 */

double
__ieee754_hypot (double x, double y)
{
  if ((isinf (x) || isinf (y))
      && !issignaling (x) && !issignaling (y))
    return INFINITY;
  if (isnan (x) || isnan (y))
    return x + y;

  x = fabs (x);
  y = fabs (y);

  if (y > x)
    {
      double t = x;
      x = y;
      y = t;
    }
  if (y == 0.0)
    return x;

  /* if y is higher enough, y * 2^60 might overflow. The tests if
     y >= 1.7976931348623157e+308/2^60 (two60factor) and uses the
     appropriate check to avoid the overflow exception generation.  */
  if (y <= 0x1.fffffffffffffp+963 && x > (y * 0x1p+60))
    return x + y;

  if (x > 0x1p+500)
    {
      x *= 0x1p-600;
      y *= 0x1p-600;
      return sqrt (x * x + y * y) / 0x1p-600;
    }
  if (y < 0x1p-500)
    {
      if (y <= 0x0.fffffffffffffp-1022)
	{
	  x *= 0x1p+1022;
	  y *= 0x1p+1022;
	  double ret = sqrt (x * x + y * y) / 0x1p+1022;
	  math_check_force_underflow_nonneg (ret);
	  return ret;
	}
      else
	{
	  x *= 0x1p+600;
	  y *= 0x1p+600;
	  return sqrt (x * x + y * y) / 0x1p+600;
	}
    }
  return sqrt (x * x + y * y);
}
#ifndef __ieee754_hypot
libm_alias_finite (__ieee754_hypot, __hypot)
#endif
