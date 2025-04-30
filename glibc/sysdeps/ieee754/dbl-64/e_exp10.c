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

static const double log10_high = 0x2.4d7637p0;
static const double log10_low = 0x7.6aaa2b05ba95cp-28;

double
__ieee754_exp10 (double arg)
{
  int32_t lx;
  double arg_high, arg_low;
  double exp_high, exp_low;

  if (!isfinite (arg))
    return __ieee754_exp (arg);
  if (arg < DBL_MIN_10_EXP - DBL_DIG - 10)
    return DBL_MIN * DBL_MIN;
  else if (arg > DBL_MAX_10_EXP + 1)
    return DBL_MAX * DBL_MAX;
  else if (fabs (arg) < 0x1p-56)
    return 1.0;

  GET_LOW_WORD (lx, arg);
  lx &= 0xf8000000;
  arg_high = arg;
  SET_LOW_WORD (arg_high, lx);
  arg_low = arg - arg_high;
  exp_high = arg_high * log10_high;
  exp_low = arg_high * log10_low + arg_low * M_LN10;
  return __ieee754_exp (exp_high) * __ieee754_exp (exp_low);
}
libm_alias_finite (__ieee754_exp10, __exp10)
