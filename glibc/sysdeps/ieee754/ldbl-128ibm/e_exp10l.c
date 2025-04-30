/* Copyright (C) 2012-2021 Free Software Foundation, Inc.
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
#include <math_private.h>
#include <float.h>
#include <libm-alias-finite.h>

static const long double log10_high = 0x2.4d763776aaap+0L;
static const long double log10_low = 0x2.b05ba95b58ae0b4c28a38a3fb4p-48L;

long double
__ieee754_exp10l (long double arg)
{
  union ibm_extended_long_double u;
  long double arg_high, arg_low;
  long double exp_high, exp_low;

  if (!isfinite (arg))
    return __ieee754_expl (arg);
  if (arg < LDBL_MIN_10_EXP - LDBL_DIG - 10)
    return LDBL_MIN * LDBL_MIN;
  else if (arg > LDBL_MAX_10_EXP + 1)
    return LDBL_MAX * LDBL_MAX;
  else if (fabsl (arg) < 0x1p-109L)
    return 1.0L;

  u.ld = arg;
  arg_high = u.d[0].d;
  arg_low = u.d[1].d;
  exp_high = arg_high * log10_high;
  exp_low = arg_high * log10_low + arg_low * M_LN10l;
  return __ieee754_expl (exp_high) * __ieee754_expl (exp_low);
}
libm_alias_finite (__ieee754_exp10l, __exp10l)
