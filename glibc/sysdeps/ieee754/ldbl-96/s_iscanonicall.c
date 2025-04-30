/* Test whether long double value is canonical.  ldbl-96 version.
   Copyright (C) 2016-2021 Free Software Foundation, Inc.
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

#include <float.h>
#include <math.h>
#include <math_private.h>
#include <stdbool.h>
#include <stdint.h>

int
__iscanonicall (long double x)
{
  uint32_t se, i0, i1 __attribute__ ((unused));

  GET_LDOUBLE_WORDS (se, i0, i1, x);
  int32_t ix = se & 0x7fff;
  bool mant_high = (i0 & 0x80000000) != 0;

  if (LDBL_MIN_EXP == -16381)
    /* Intel variant: the high mantissa bit should have a value
       determined by the exponent.  */
    return ix > 0 ? mant_high : !mant_high;
  else
    /* M68K variant: both values of the high bit are valid for the
       greatest and smallest exponents, while other exponents require
       the high bit to be set.  */
    return ix == 0 || ix == 0x7fff || mant_high;
}
libm_hidden_def (__iscanonicall)
