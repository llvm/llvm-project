/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
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
   License along with the GNU C Library.  If not, see
   <https://www.gnu.org/licenses/>.  */

#include <math.h>
#include <math_private.h>
#include "mathimpl.h"
#include <libm-alias-finite.h>

#ifndef SUFF
#define SUFF
#endif
#ifndef float_type
#define float_type double
#endif

#define CONCATX(a,b) __CONCAT(a,b)
#define s(name) CONCATX(name,SUFF)
#define m81(func) __m81_u(s(func))

float_type
s(__ieee754_atan2) (float_type y, float_type x)
{
  float_type pi, pi_2, z;
  unsigned long y_cond, x_cond;

  __asm ("fmovecr%.x %#0, %0" : "=f" (pi));
  __asm ("fscale%.w %#-1, %0" : "=f" (pi_2) : "0" (pi));
  y_cond = __m81_test (y);
  x_cond = __m81_test (x);

  if ((x_cond | y_cond) & __M81_COND_NAN)
    z = x + y;
  else if (y_cond & __M81_COND_ZERO)
    {
      if (x_cond & __M81_COND_NEG)
	z = y_cond & __M81_COND_NEG ? -pi : pi;
      else
	z = y;
    }
  else if (x_cond & __M81_COND_INF)
    {
      if (y_cond & __M81_COND_INF)
	{
	  float_type pi_4;
	  __asm ("fscale%.w %#-2, %0" : "=f" (pi_4) : "0" (pi));
	  z = x_cond & __M81_COND_NEG ? 3 * pi_4 : pi_4;
	}
      else
	z = x_cond & __M81_COND_NEG ? pi : 0;
      if (y_cond & __M81_COND_NEG)
	z = -z;
    }
  else if (y_cond & __M81_COND_INF)
    z = y_cond & __M81_COND_NEG ? -pi_2 : pi_2;
  else if (x_cond & __M81_COND_NEG)
    {
      if (y_cond & __M81_COND_NEG)
	{
	  if (-x > -y)
	    z = -pi + m81(__atan) (y / x);
	  else
	    z = -pi_2 - m81(__atan) (x / y);
	}
      else
	{
	  if (-x > y)
	    z = pi + m81(__atan) (y / x);
	  else
	    z = pi_2 - m81(__atan) (x / y);
	}
    }
  else
    {
      if (y_cond & __M81_COND_NEG)
	{
	  if (x > -y)
	    z = m81(__atan) (y / x);
	  else
	    z = -pi_2 - m81(__atan) (x / y);
	}
      else
	{
	  if (x > y)
	    z = m81(__atan) (y / x);
	  else
	    z = pi_2 - m81(__atan) (x / y);
	}
    }
  return z;
}
libm_alias_finite (s(__ieee754_atan2), s (__atan2))
