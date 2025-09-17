//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/float/definitions.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_exp.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_lgamma.h>
#include <clc/math/clc_sinpi.h>
#include <clc/math/math.h>

_CLC_OVERLOAD _CLC_DEF float __clc_tgamma(float x) {
  const float pi = 3.1415926535897932384626433832795f;
  float absx = __clc_fabs(x);
  float lg = __clc_lgamma(absx);
  float g = __clc_exp(lg);

  if (x < 0.0f) {
    float z = __clc_sinpi(x);
    g = g * absx * z;
    g = pi / g;
    g = g == 0 ? INFINITY : g;
    g = z == 0 ? FLT_NAN : g;
  }

  return g;
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_OVERLOAD _CLC_DEF double __clc_tgamma(double x) {
  const double pi = 3.1415926535897932384626433832795;
  double absx = __clc_fabs(x);
  double lg = __clc_lgamma(absx);
  double g = __clc_exp(lg);

  if (x < 0.0) {
    double z = __clc_sinpi(x);
    g = g * absx * z;
    g = pi / g;
    g = g == 0 ? INFINITY : g;
    g = z == 0 ? DBL_NAN : g;
  }

  return g;
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Forward the half version of this builtin onto the float one
_CLC_OVERLOAD _CLC_DEF half __clc_tgamma(half x) {
  return (half)__clc_tgamma((float)x);
}

#endif

#define __CLC_FUNCTION __clc_tgamma
#define __CLC_BODY <clc/shared/unary_def_scalarize.inc>
#include <clc/math/gentype.inc>
