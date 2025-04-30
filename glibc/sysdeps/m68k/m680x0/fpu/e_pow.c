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
s(__ieee754_pow) (float_type x, float_type y)
{
  float_type z;
  float_type ax;
  unsigned long x_cond, y_cond;

  y_cond = __m81_test (y);
  if (y_cond & __M81_COND_ZERO)
    return 1.0;
  if (y_cond & __M81_COND_NAN)
    return x == 1.0 ? x : x + y;

  x_cond = __m81_test (x);
  if (x_cond & __M81_COND_NAN)
    return x + y;

  if (y_cond & __M81_COND_INF)
    {
      ax = s(fabs) (x);
      if (ax == 1.0)
	return ax;
      if (ax > 1.0)
	return y_cond & __M81_COND_NEG ? 0 : y;
      else
	return y_cond & __M81_COND_NEG ? -y : 0;
    }

  if (s(fabs) (y) == 1.0)
    return y_cond & __M81_COND_NEG ? 1 / x : x;

  if (y == 2)
    return x * x;
  if (y == 0.5 && !(x_cond & __M81_COND_NEG))
    return m81(sqrt) (x);

  if (x == 10.0)
    {
      __asm ("ftentox%.x %1, %0" : "=f" (z) : "f" (y));
      return z;
    }
  if (x == 2.0)
    {
      __asm ("ftwotox%.x %1, %0" : "=f" (z) : "f" (y));
      return z;
    }

  ax = s(fabs) (x);
  if (x_cond & (__M81_COND_INF | __M81_COND_ZERO) || ax == 1.0)
    {
      z = ax;
      if (y_cond & __M81_COND_NEG)
	z = 1 / z;
      if (x_cond & __M81_COND_NEG)
	{
	  if (y != m81(__rint) (y))
	    {
	      if (x == -1)
		z = (z - z) / (z - z);
	    }
	  else
	    goto maybe_negate;
	}
      return z;
    }

  if (x_cond & __M81_COND_NEG)
    {
      if (y == m81(__rint) (y))
	{
	  z = m81(__ieee754_exp) (y * m81(__ieee754_log) (-x));
	maybe_negate:
	  /* We always use the long double format, since y is already in
	     this format and rounding won't change the result.  */
	  {
	    int32_t exponent;
	    uint32_t i0, i1;
	    GET_LDOUBLE_WORDS (exponent, i0, i1, y);
	    exponent = (exponent & 0x7fff) - 0x3fff;
	    if (exponent <= 31
		? i0 & (1 << (31 - exponent))
		: (exponent <= 63
		   && i1 & (1 << (63 - exponent))))
	      z = -z;
	  }
	}
      else
	z = (y - y) / (y - y);
    }
  else
    z = m81(__ieee754_exp) (y * m81(__ieee754_log) (x));
  return z;
}
libm_alias_finite (s(__ieee754_pow), s (__pow))
