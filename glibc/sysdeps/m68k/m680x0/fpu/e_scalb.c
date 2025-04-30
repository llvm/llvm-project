/* Copyright (C) 1997-2021 Free Software Foundation, Inc.
   This file is part of the GNU C Library.
   Contributed by Andreas Schwab <schwab@issan.informatik.uni-dortmund.de>.

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
s(__ieee754_scalb) (float_type x, float_type fn)
{
  float_type retval;
  unsigned long x_cond = __m81_test (x);
  unsigned long fn_cond = __m81_test (fn);

  if ((x_cond | fn_cond) & __M81_COND_NAN)
    return x * fn;

  if (fn_cond & __M81_COND_INF)
    {
      if (!(fn_cond & __M81_COND_NEG))
	return x * fn;
      else if (x_cond & __M81_COND_ZERO)
	return x;
      else
	return x / -fn;
    }

  if (m81(__rint) (fn) != fn)
    return (x - x) / (x - x);

  __asm ("fscale%.x %1, %0" : "=f" (retval) : "f" (fn), "0" (x));
  return retval;
}
libm_alias_finite (s(__ieee754_scalb), s (__scalb))
