/* Single-precision 2^x function.
   Copyright (C) 2017-2021 Free Software Foundation, Inc.
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
#include <math-narrow-eval.h>
#include <stdint.h>
#include <libm-alias-finite.h>
#include <libm-alias-float.h>
#include "math_config.h"

/*
EXP2F_TABLE_BITS = 5
EXP2F_POLY_ORDER = 3

ULP error: 0.502 (nearest rounding.)
Relative error: 1.69 * 2^-34 in [-1/64, 1/64] (before rounding.)
Wrong count: 168353 (all nearest rounding wrong results with fma.)
Non-nearest ULP error: 1 (rounded ULP error)
*/

#define N (1 << EXP2F_TABLE_BITS)
#define T __exp2f_data.tab
#define C __exp2f_data.poly
#define SHIFT __exp2f_data.shift_scaled

static inline uint32_t
top12 (float x)
{
  return asuint (x) >> 20;
}

float
__exp2f (float x)
{
  uint32_t abstop;
  uint64_t ki, t;
  /* double_t for better performance on targets with FLT_EVAL_METHOD==2.  */
  double_t kd, xd, z, r, r2, y, s;

  xd = (double_t) x;
  abstop = top12 (x) & 0x7ff;
  if (__glibc_unlikely (abstop >= top12 (128.0f)))
    {
      /* |x| >= 128 or x is nan.  */
      if (asuint (x) == asuint (-INFINITY))
	return 0.0f;
      if (abstop >= top12 (INFINITY))
	return x + x;
      if (x > 0.0f)
	return __math_oflowf (0);
      if (x <= -150.0f)
	return __math_uflowf (0);
#if WANT_ERRNO_UFLOW
      if (x < -149.0f)
	return __math_may_uflowf (0);
#endif
    }

  /* x = k/N + r with r in [-1/(2N), 1/(2N)] and int k.  */
  kd = math_narrow_eval ((double) (xd + SHIFT)); /* Needs to be double.  */
  ki = asuint64 (kd);
  kd -= SHIFT; /* k/N for int k.  */
  r = xd - kd;

  /* exp2(x) = 2^(k/N) * 2^r ~= s * (C0*r^3 + C1*r^2 + C2*r + 1) */
  t = T[ki % N];
  t += ki << (52 - EXP2F_TABLE_BITS);
  s = asdouble (t);
  z = C[0] * r + C[1];
  r2 = r * r;
  y = C[2] * r + 1;
  y = z * r2 + y;
  y = y * s;
  return (float) y;
}
#ifndef __exp2f
strong_alias (__exp2f, __ieee754_exp2f)
libm_alias_finite (__ieee754_exp2f, __exp2f)
versioned_symbol (libm, __exp2f, exp2f, GLIBC_2_27);
libm_alias_float_other (__exp2, exp2)
#endif
