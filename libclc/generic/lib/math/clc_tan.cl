//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sincos_helpers.h"
#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_sincos_helpers.h>
#include <clc/math/math.h>
#include <clc/math/tables.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>

_CLC_DEF _CLC_OVERLOAD float __clc_tan(float x) {
  int ix = as_int(x);
  int ax = ix & 0x7fffffff;
  float dx = as_float(ax);

  float r0, r1;
  int regn = __clc_argReductionS(&r0, &r1, dx);

  float t = __clc_tanf_piby4(r0 + r1, regn);
  t = as_float(as_int(t) ^ (ix ^ ax));

  t = ax >= PINFBITPATT_SP32 ? as_float(QNANBITPATT_SP32) : t;
  // Take care of subnormals
  t = (x == 0.0f) ? x : t;
  return t;
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, float, __clc_tan, float);

#ifdef cl_khr_fp64
#include <clc/math/clc_sincos_piby4.h>

_CLC_DEF _CLC_OVERLOAD double __clc_tan(double x) {
  double y = __clc_fabs(x);

  double r, rr;
  int regn;

  if (y < 0x1.0p+30)
    __clc_remainder_piby2_medium(y, &r, &rr, &regn);
  else
    __clc_remainder_piby2_large(y, &r, &rr, &regn);

  double lead, tail;
  __clc_tan_piby4(r, rr, &lead, &tail);

  int2 t = as_int2(regn & 1 ? tail : lead);
  t.hi ^= (x < 0.0) << 31;

  return __clc_isnan(x) || __clc_isinf(x) ? as_double(QNANBITPATT_DP64)
                                          : as_double(t);
}
_CLC_UNARY_VECTORIZE(_CLC_DEF _CLC_OVERLOAD, double, __clc_tan, double);

#endif
