//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/float/definitions.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_copysign.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_frexp.h>
#include <clc/math/clc_ldexp.h>
#include <clc/math/clc_recip_fast.h>
#include <clc/math/clc_rint.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isfinite.h>
#include <clc/relational/clc_isnan.h>

_CLC_DEF _CLC_OVERLOAD float __clc_fmod(float x, float y) {
  // How many bits of the quotient to resolve per iteration.
  const int bits = 12;

  float ax = __clc_fabs(x);
  float ay = __clc_fabs(y);

  float ret;

  if (ax > ay) {
    int ex, ey;
    float mx = __clc_frexp(ax, &ex);
    --ex;
    float my = __clc_frexp(ay, &ey);
    --ey;

    ax = __clc_ldexp(mx, bits);
    ay = __clc_ldexp(my, 1);

    int nb = ex - ey;
    float ayinv = __clc_recip_fast(ay);

    while (nb > bits) {
      float q = __clc_rint(ax * ayinv);
      ax = __clc_fma(-q, ay, ax);
      int clt = ax < 0.0f;
      float axp = ax + ay;
      ax = clt ? axp : ax;
      ax = __clc_ldexp(ax, bits);
      nb -= bits;
    }

    ax = __clc_ldexp(ax, nb - bits + 1);

    // Final iteration.
    float q = __clc_rint(ax * ayinv);
    ax = __clc_fma(-q, ay, ax);
    int clt = ax < 0.0f;
    float axp = ax + ay;
    ax = clt ? axp : ax;

    ax = __clc_ldexp(ax, ey);
    ret = __clc_as_float((__clc_as_int(x) & SIGNBIT_SP32) ^ __clc_as_int(ax));
  } else {
    // |x| < |y| returns x; |x| == |y| returns a zero with the sign of x.
    ret = ax == ay ? __clc_copysign(0.0f, x) : x;
  }

  // fmod(x, 0) is NaN; fmod(Inf, y) is NaN; fmod(x, NaN)/fmod(NaN, y) is NaN.
  ret = y == 0.0f ? FLT_NAN : ret;
  int c = !__clc_isnan(y) && __clc_isfinite(x);
  ret = c ? ret : FLT_NAN;

  return ret;
}

#define __CLC_FLOAT_ONLY
#define __CLC_FUNCTION __clc_fmod
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_FUNCTION

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __clc_fmod(double x, double y) {
  // How many bits of the quotient to resolve per iteration.
  const int bits = 26;

  double ax = __clc_fabs(x);
  double ay = __clc_fabs(y);

  double ret;

  if (ax > ay) {
    int ex, ey;
    double mx = __clc_frexp(ax, &ex);
    --ex;
    double my = __clc_frexp(ay, &ey);
    --ey;

    ax = __clc_ldexp(mx, bits);
    ay = __clc_ldexp(my, 1);

    int nb = ex - ey;
    double ayinv = 1.0 / ay;

    while (nb > bits) {
      double q = __clc_rint(ax * ayinv);
      ax = __clc_fma(-q, ay, ax);
      int clt = ax < 0.0;
      double axp = ax + ay;
      ax = clt ? axp : ax;
      ax = __clc_ldexp(ax, bits);
      nb -= bits;
    }

    ax = __clc_ldexp(ax, nb - bits + 1);

    // Final iteration.
    double q = __clc_rint(ax * ayinv);
    ax = __clc_fma(-q, ay, ax);
    int clt = ax < 0.0;
    double axp = ax + ay;
    ax = clt ? axp : ax;

    ax = __clc_ldexp(ax, ey);
    ret = __clc_as_double((__clc_as_ulong(x) & SIGNBIT_DP64) ^
                          __clc_as_ulong(ax));
  } else {
    // |x| < |y| returns x; |x| == |y| returns a zero with the sign of x.
    ret = ax == ay ? __clc_copysign(0.0, x) : x;
  }

  // fmod(x, 0) is NaN; fmod(Inf, y) is NaN; fmod(x, NaN)/fmod(NaN, y) is NaN.
  ret = y == 0.0 ? DBL_NAN : ret;
  int c = !__clc_isnan(y) && __clc_isfinite(x);
  ret = c ? ret : DBL_NAN;

  return ret;
}

#define __CLC_DOUBLE_ONLY
#define __CLC_FUNCTION __clc_fmod
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_FUNCTION

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Forward the half version of this builtin onto the float one
#define __CLC_HALF_ONLY
#define __CLC_FUNCTION __clc_fmod
#define __CLC_BODY <clc/math/binary_def_via_fp32.inc>
#include <clc/math/gentype.inc>

#endif
