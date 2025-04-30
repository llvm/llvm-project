/* Round to nearest integer.  PowerPC64 version.
   Copyright (C) 2019-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library

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

#define NO_MATH_REDIRECT
#define lroundf __redirect_llround
#define __lroundf __redirect___lround
#include <math.h>
#undef lroundf
#undef __lroundf
#include <libm-alias-float.h>
#include <math-barriers.h>

long long int
__llroundf (float x)
{
#ifdef _ARCH_PWR5X
  double r = __builtin_round (x);
  /* Prevent gcc from calling llround directly when compiled with
     -fno-math-errno by inserting a barrier.  */
  math_opt_barrier (r);
  return r;
#else
   /* IEEE 1003.1 llroundf function.  IEEE specifies "roundf to the nearest
      integer value, rounding halfway cases away from zero, regardless of
      the current rounding mode."  However PowerPC Architecture defines
      "roundf to Nearest" as "Choose the best approximation. In case of a
      tie, choose the one that is even (least significant bit o).".
      So we can't use the PowerPC "round to Nearest" mode. Instead we set
      "round toward Zero" mode and round by adding +-0.5 before rounding
      to the integer value.

      It is necessary to detect when x is (+-)0x1.fffffffffffffp-2
      because adding +-0.5 in this case will cause an erroneous shift,
      carry and round.  We simply return 0 if 0.5 > x > -0.5.  Likewise
      if x is and odd number between +-(2^23 and 2^24-1) a shift and
      carry will erroneously round if biased with +-0.5.  Therefore if x
      is greater/less than +-2^23 we don't need to bias the number with
      +-0.5.  */

  float ax = fabsf (x);

  if (ax < 0.5f)
    return 0;

  if (ax < 0x1p+23f)
    {
      /* Test whether an integer to avoid spurious "inexact".  */
      float t = ax + 0x1p+23f;
      t = t - 0x1p+23f;
      if (ax != t)
	{
	  ax = ax + 0.5f;
	  if (x < 0.0f)
	    ax = -fabs (ax);
	  x = ax;
	}
    }

  long int ret;
  __asm__ ("fctidz %0, %1" : "=d" (ret) : "d" (x));
  return ret;
#endif
}
#ifndef __llroundf
strong_alias (__llroundf, __lroundf)
libm_alias_float (__llround, lround)
libm_alias_float (__llround, llround)
#endif
