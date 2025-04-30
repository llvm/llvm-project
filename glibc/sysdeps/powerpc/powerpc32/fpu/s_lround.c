/* lround function.  PowerPC32 version.
   Copyright (C) 2004-2021 Free Software Foundation, Inc.
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

#define lroundf __redirect_lroundf
#define __lroundf __redirect___lroundf
#include <math.h>
#undef lroundf
#undef __lroundf
#include <libm-alias-float.h>
#include <libm-alias-double.h>

long int
__lround (double x)
{
#ifdef _ARCH_PWR5X
  x = round (x);
#else
  /* Ieee 1003.1 lround function.  ieee specifies "round to the nearest
     integer value, rounding halfway cases away from zero, regardless of
     the current rounding mode."  however powerpc architecture defines
     "round to nearest" as "choose the best approximation. in case of a
     tie, choose the one that is even (least significant bit o).".
     so we can't use the powerpc "round to nearest" mode. instead we set
     "round toward zero" mode and round by adding +-0.5 before rounding
     to the integer value.  it is necessary to detect when x is
     (+-)0x1.fffffffffffffp-2 because adding +-0.5 in this case will
     cause an erroneous shift, carry and round.  we simply return 0 if
     0.5 > x > -0.5.  */

  double ax = fabs (x);

  if (ax < 0.5)
    return 0;

  if (x >= 0x7fffffff.8p0 || x <= -0x80000000.8p0)
    x = (x < 0.0) ? -0x1p+52 : 0x1p+52;
  else
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
#endif
  /* Force evaluation of values larger than long int, so invalid
     exceptions are raise.  */
  long long int ret;
  asm ("fctiwz %0, %1" : "=d" (ret) : "d" (x));
  return ret;
}
#ifndef __lround
libm_alias_double (__lround, lround)

strong_alias (__lround, __lroundf)
libm_alias_float (__lround, lround)
#endif
