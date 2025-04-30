/* Double-precision e^x function.
   Copyright (C) 2018-2021 Free Software Foundation, Inc.
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
#include <stdint.h>
#include <math-barriers.h>
#include <math-narrow-eval.h>
#include <math-svid-compat.h>
#include <libm-alias-finite.h>
#include <libm-alias-double.h>
#include "math_config.h"

#define N (1 << EXP_TABLE_BITS)
#define InvLn2N __exp_data.invln2N
#define NegLn2hiN __exp_data.negln2hiN
#define NegLn2loN __exp_data.negln2loN
#define Shift __exp_data.shift
#define T __exp_data.tab
#define C2 __exp_data.poly[5 - EXP_POLY_ORDER]
#define C3 __exp_data.poly[6 - EXP_POLY_ORDER]
#define C4 __exp_data.poly[7 - EXP_POLY_ORDER]
#define C5 __exp_data.poly[8 - EXP_POLY_ORDER]

/* Handle cases that may overflow or underflow when computing the result that
   is scale*(1+TMP) without intermediate rounding.  The bit representation of
   scale is in SBITS, however it has a computed exponent that may have
   overflown into the sign bit so that needs to be adjusted before using it as
   a double.  (int32_t)KI is the k used in the argument reduction and exponent
   adjustment of scale, positive k here means the result may overflow and
   negative k means the result may underflow.  */
static inline double
specialcase (double_t tmp, uint64_t sbits, uint64_t ki)
{
  double_t scale, y;

  if ((ki & 0x80000000) == 0)
    {
      /* k > 0, the exponent of scale might have overflowed by <= 460.  */
      sbits -= 1009ull << 52;
      scale = asdouble (sbits);
      y = 0x1p1009 * (scale + scale * tmp);
      return check_oflow (y);
    }
  /* k < 0, need special care in the subnormal range.  */
  sbits += 1022ull << 52;
  scale = asdouble (sbits);
  y = scale + scale * tmp;
  if (y < 1.0)
    {
      /* Round y to the right precision before scaling it into the subnormal
	 range to avoid double rounding that can cause 0.5+E/2 ulp error where
	 E is the worst-case ulp error outside the subnormal range.  So this
	 is only useful if the goal is better than 1 ulp worst-case error.  */
      double_t hi, lo;
      lo = scale - y + scale * tmp;
      hi = 1.0 + y;
      lo = 1.0 - hi + y + lo;
      y = math_narrow_eval (hi + lo) - 1.0;
      /* Avoid -0.0 with downward rounding.  */
      if (WANT_ROUNDING && y == 0.0)
	y = 0.0;
      /* The underflow exception needs to be signaled explicitly.  */
      math_force_eval (math_opt_barrier (0x1p-1022) * 0x1p-1022);
    }
  y = 0x1p-1022 * y;
  return check_uflow (y);
}

/* Top 12 bits of a double (sign and exponent bits).  */
static inline uint32_t
top12 (double x)
{
  return asuint64 (x) >> 52;
}

#ifndef SECTION
# define SECTION
#endif

double
SECTION
__exp (double x)
{
  uint32_t abstop;
  uint64_t ki, idx, top, sbits;
  /* double_t for better performance on targets with FLT_EVAL_METHOD==2.  */
  double_t kd, z, r, r2, scale, tail, tmp;

  abstop = top12 (x) & 0x7ff;
  if (__glibc_unlikely (abstop - top12 (0x1p-54)
			>= top12 (512.0) - top12 (0x1p-54)))
    {
      if (abstop - top12 (0x1p-54) >= 0x80000000)
	/* Avoid spurious underflow for tiny x.  */
	/* Note: 0 is common input.  */
	return WANT_ROUNDING ? 1.0 + x : 1.0;
      if (abstop >= top12 (1024.0))
	{
	  if (asuint64 (x) == asuint64 (-INFINITY))
	    return 0.0;
	  if (abstop >= top12 (INFINITY))
	    return 1.0 + x;
	  if (asuint64 (x) >> 63)
	    return __math_uflow (0);
	  else
	    return __math_oflow (0);
	}
      /* Large x is special cased below.  */
      abstop = 0;
    }

  /* exp(x) = 2^(k/N) * exp(r), with exp(r) in [2^(-1/2N),2^(1/2N)].  */
  /* x = ln2/N*k + r, with int k and r in [-ln2/2N, ln2/2N].  */
  z = InvLn2N * x;
#if TOINT_INTRINSICS
  kd = roundtoint (z);
  ki = converttoint (z);
#else
  /* z - kd is in [-1, 1] in non-nearest rounding modes.  */
  kd = math_narrow_eval (z + Shift);
  ki = asuint64 (kd);
  kd -= Shift;
#endif
  r = x + kd * NegLn2hiN + kd * NegLn2loN;
  /* 2^(k/N) ~= scale * (1 + tail).  */
  idx = 2 * (ki % N);
  top = ki << (52 - EXP_TABLE_BITS);
  tail = asdouble (T[idx]);
  /* This is only a valid scale when -1023*N < k < 1024*N.  */
  sbits = T[idx + 1] + top;
  /* exp(x) = 2^(k/N) * exp(r) ~= scale + scale * (tail + exp(r) - 1).  */
  /* Evaluation is optimized assuming superscalar pipelined execution.  */
  r2 = r * r;
  /* Without fma the worst case error is 0.25/N ulp larger.  */
  /* Worst case error is less than 0.5+1.11/N+(abs poly error * 2^53) ulp.  */
  tmp = tail + r + r2 * (C2 + r * C3) + r2 * r2 * (C4 + r * C5);
  if (__glibc_unlikely (abstop == 0))
    return specialcase (tmp, sbits, ki);
  scale = asdouble (sbits);
  /* Note: tmp == 0 or |tmp| > 2^-65 and scale > 2^-739, so there
     is no spurious underflow here even without fma.  */
  return scale + scale * tmp;
}
#ifndef __exp
hidden_def (__exp)
strong_alias (__exp, __ieee754_exp)
libm_alias_finite (__ieee754_exp, __exp)
# if LIBM_SVID_COMPAT
versioned_symbol (libm, __exp, exp, GLIBC_2_29);
libm_alias_double_other (__exp, exp)
# else
libm_alias_double (__exp, exp)
# endif
#endif
