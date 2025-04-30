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

static const _Float128 log10_high = L(0x2.4d763776aaa2bp0);
static const _Float128 log10_low = L(0x5.ba95b58ae0b4c28a38a3fb3e7698p-60);

_Float128
__ieee754_exp10l (_Float128 arg)
{
  ieee854_long_double_shape_type u;
  _Float128 arg_high, arg_low;
  _Float128 exp_high, exp_low;

  if (!isfinite (arg))
    return __ieee754_expl (arg);
  if (arg < LDBL_MIN_10_EXP - LDBL_DIG - 10)
    return LDBL_MIN * LDBL_MIN;
  else if (arg > LDBL_MAX_10_EXP + 1)
    return LDBL_MAX * LDBL_MAX;
  else if (fabsl (arg) < L(0x1p-116))
    return 1;

  u.value = arg;
  u.parts64.lsw &= 0xfe00000000000000LL;
  arg_high = u.value;
  arg_low = arg - arg_high;
  exp_high = arg_high * log10_high;
  exp_low = arg_high * log10_low + arg_low * M_LN10l;
  return __ieee754_expl (exp_high) * __ieee754_expl (exp_low);
}
libm_alias_finite (__ieee754_exp10l, __exp10l)
