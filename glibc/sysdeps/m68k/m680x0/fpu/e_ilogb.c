/* Copyright (C) 1996-2021 Free Software Foundation, Inc.
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
#include "mathimpl.h"

#ifndef SUFF
#define SUFF
#endif
#ifndef float_type
#define float_type double
#endif

#define CONCATX(a,b) __CONCAT(a,b)
#define s(name) CONCATX(name,SUFF)
#define m81(func) __m81_u(s(func))

int
s(__ieee754_ilogb) (float_type x)
{
  float_type result;
  unsigned long x_cond;

  x_cond = __m81_test (x);
  /* We must return consistent values for zero and NaN.  */
  if (x_cond & __M81_COND_ZERO)
    return FP_ILOGB0;
  if (x_cond & (__M81_COND_NAN | __M81_COND_INF))
    return FP_ILOGBNAN;

  __asm ("fgetexp%.x %1, %0" : "=f" (result) : "f" (x));
  return (int) result;
}
