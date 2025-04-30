/* Single-precision 10^x function.
   Copyright (C) 2020-2021 Free Software Foundation, Inc.
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
#include <shlib-compat.h>
#include <math-svid-compat.h>
#include "math_config.h"

/*
  EXP2F_TABLE_BITS 5
  EXP2F_POLY_ORDER 3

  Max. ULP error: 0.502 (normal range, nearest rounding).
  Max. relative error: 2^-33.240 (before rounding to float).
  Wrong count: 169839.
  Non-nearest ULP error: 1 (rounded ULP error).

  Detailed error analysis (for EXP2F_TABLE_BITS=3 thus N=32):

  - We first compute z = RN(InvLn10N * x) where

      InvLn10N = N*log(10)/log(2) * (1 + theta1) with |theta1| < 2^-54.150

    since z is rounded to nearest:

      z = InvLn10N * x * (1 + theta2) with |theta2| < 2^-53

    thus z =  N*log(10)/log(2) * x * (1 + theta3) with |theta3| < 2^-52.463

  - Since |x| < 38 in the main branch, we deduce:

    z = N*log(10)/log(2) * x + theta4 with |theta4| < 2^-40.483

  - We then write z = k + r where k is an integer and |r| <= 0.5 (exact)

  - We thus have

    x = z*log(2)/(N*log(10)) - theta4*log(2)/(N*log(10))
      = z*log(2)/(N*log(10)) + theta5 with |theta5| < 2^-47.215

    10^x = 2^(k/N) * 2^(r/N) * 10^theta5
         =  2^(k/N) * 2^(r/N) * (1 + theta6) with |theta6| < 2^-46.011

  - Then 2^(k/N) is approximated by table lookup, the maximal relative error
    is for (k%N) = 5, with

      s = 2^(i/N) * (1 + theta7) with |theta7| < 2^-53.249

  - 2^(r/N) is approximated by a degree-3 polynomial, where the maximal
    mathematical relative error is 2^-33.243.

  - For C[0] * r + C[1], assuming no FMA is used, since |r| <= 0.5 and
    |C[0]| < 1.694e-6, |C[0] * r| < 8.47e-7, and the absolute error on
    C[0] * r is bounded by 1/2*ulp(8.47e-7) = 2^-74.  Then we add C[1] with
    |C[1]| < 0.000235, thus the absolute error on C[0] * r + C[1] is bounded
    by 2^-65.994 (z is bounded by 0.000236).

  - For r2 = r * r, since |r| <= 0.5, we have |r2| <= 0.25 and the absolute
    error is bounded by 1/4*ulp(0.25) = 2^-56 (the factor 1/4 is because |r2|
    cannot exceed 1/4, and on the left of 1/4 the distance between two
    consecutive numbers is 1/4*ulp(1/4)).

  - For y = C[2] * r + 1, assuming no FMA is used, since |r| <= 0.5 and
    |C[2]| < 0.0217, the absolute error on C[2] * r is bounded by 2^-60,
    and thus the absolute error on C[2] * r + 1 is bounded by 1/2*ulp(1)+2^60
    < 2^-52.988, and |y| < 1.01085 (the error bound is better if a FMA is
    used).

  - for z * r2 + y: the absolute error on z is bounded by 2^-65.994, with
    |z| < 0.000236, and the absolute error on r2 is bounded by 2^-56, with
    r2 < 0.25, thus |z*r2| < 0.000059, and the absolute error on z * r2
    (including the rounding error) is bounded by:

      2^-65.994 * 0.25 + 0.000236 * 2^-56 + 1/2*ulp(0.000059) < 2^-66.429.

    Now we add y, with |y| < 1.01085, and error on y bounded by 2^-52.988,
    thus the absolute error is bounded by:

      2^-66.429 + 2^-52.988 + 1/2*ulp(1.01085) < 2^-51.993

  - Now we convert the error on y into relative error.  Recall that y
    approximates 2^(r/N), for |r| <= 0.5 and N=32. Thus 2^(-0.5/32) <= y,
    and the relative error on y is bounded by

      2^-51.993/2^(-0.5/32) < 2^-51.977

  - Taking into account the mathematical relative error of 2^-33.243 we have:

      y = 2^(r/N) * (1 + theta8) with |theta8| < 2^-33.242

  - Since we had s = 2^(k/N) * (1 + theta7) with |theta7| < 2^-53.249,
    after y = y * s we get y = 2^(k/N) * 2^(r/N) * (1 + theta9)
    with |theta9| < 2^-33.241

  - Finally, taking into account the error theta6 due to the rounding error on
    InvLn10N, and when multiplying InvLn10N * x, we get:

      y = 10^x * (1 + theta10) with |theta10| < 2^-33.240

  - Converting into binary64 ulps, since y < 2^53*ulp(y), the error is at most
    2^19.76 ulp(y)

  - If the result is a binary32 value in the normal range (i.e., >= 2^-126),
    then the error is at most 2^-9.24 ulps, i.e., less than 0.00166 (in the
    subnormal range, the error in ulps might be larger).

  Note that this bound is tight, since for x=-0x25.54ac0p0 the final value of
  y (before conversion to float) differs from 879853 ulps from the correctly
  rounded value, and 879853 ~ 2^19.7469.  */

#define N (1 << EXP2F_TABLE_BITS)

#define InvLn10N (0x3.5269e12f346e2p0 * N) /* log(10)/log(2) to nearest */
#define T __exp2f_data.tab
#define C __exp2f_data.poly_scaled
#define SHIFT __exp2f_data.shift

static inline uint32_t
top13 (float x)
{
  return asuint (x) >> 19;
}

float
__exp10f (float x)
{
  uint32_t abstop;
  uint64_t ki, t;
  double kd, xd, z, r, r2, y, s;

  xd = (double) x;
  abstop = top13 (x) & 0xfff; /* Ignore sign.  */
  if (__glibc_unlikely (abstop >= top13 (38.0f)))
    {
      /* |x| >= 38 or x is nan.  */
      if (asuint (x) == asuint (-INFINITY))
        return 0.0f;
      if (abstop >= top13 (INFINITY))
        return x + x;
      /* 0x26.8826ap0 is the largest value such that 10^x < 2^128.  */
      if (x > 0x26.8826ap0f)
        return __math_oflowf (0);
      /* -0x2d.278d4p0 is the smallest value such that 10^x > 2^-150.  */
      if (x < -0x2d.278d4p0f)
        return __math_uflowf (0);
#if WANT_ERRNO_UFLOW
      if (x < -0x2c.da7cfp0)
        return __math_may_uflowf (0);
#endif
      /* the smallest value such that 10^x >= 2^-126 (normal range)
         is x = -0x25.ee060p0 */
      /* we go through here for 2014929 values out of 2060451840
         (not counting NaN and infinities, i.e., about 0.1% */
    }

  /* x*N*Ln10/Ln2 = k + r with r in [-1/2, 1/2] and int k.  */
  z = InvLn10N * xd;
  /* |xd| < 38 thus |z| < 1216 */
#if TOINT_INTRINSICS
  kd = roundtoint (z);
  ki = converttoint (z);
#else
# define SHIFT __exp2f_data.shift
  kd = math_narrow_eval ((double) (z + SHIFT)); /* Needs to be double.  */
  ki = asuint64 (kd);
  kd -= SHIFT;
#endif
  r = z - kd;

  /* 10^x = 10^(k/N) * 10^(r/N) ~= s * (C0*r^3 + C1*r^2 + C2*r + 1)  */
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
#ifndef __exp10f
strong_alias (__exp10f, __ieee754_exp10f)
libm_alias_finite (__ieee754_exp10f, __exp10f)
/* For architectures that already provided exp10f without SVID support, there
   is no need to add a new version.  */
#if !LIBM_SVID_COMPAT
# define EXP10F_VERSION GLIBC_2_26
#else
# define EXP10F_VERSION GLIBC_2_32
#endif
versioned_symbol (libm, __exp10f, exp10f, EXP10F_VERSION);
libm_alias_float_other (__exp10, exp10)
#endif
