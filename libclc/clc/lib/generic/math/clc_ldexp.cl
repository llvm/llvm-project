//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/integer/clc_add_sat.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_subnormal_config.h>
#include <clc/math/math.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>
#include <clc/shared/clc_clamp.h>

_CLC_DEF _CLC_OVERLOAD float __clc_ldexp(float x, int n) {

  if (!__clc_fp32_subnormals_supported()) {
    // This treats subnormals as zeros
    int i = __clc_as_int(x);
    int e = (i >> 23) & 0xff;
    int m = i & 0x007fffff;
    int s = i & 0x80000000;
    int v = __clc_add_sat(e, n);
    v = __clc_clamp(v, 0, 0xff);
    int mr = (e == 0 || v == 0 || v == 0xff) ? 0 : m;
    int c = e == 0xff;
    mr = c ? m : mr;
    int er = c ? e : v;
    er = e ? er : e;
    return __clc_as_float(s | (er << 23) | mr);
  }

  /* supports denormal values */
  const int multiplier = 24;
  float val_f;
  uint val_ui;
  uint sign;
  int exponent;
  val_ui = __clc_as_uint(x);
  sign = val_ui & 0x80000000;
  val_ui = val_ui & 0x7fffffff; /* remove the sign bit */
  int val_x = val_ui;

  exponent = val_ui >> 23; /* get the exponent */
  int dexp = exponent;

  /* denormal support */
  int fbh =
      127 -
      (__clc_as_uint((float)(__clc_as_float(val_ui | 0x3f800000) - 1.0f)) >>
       23);
  int dexponent = 25 - fbh;
  uint dval_ui = (((val_ui << fbh) & 0x007fffff) | (dexponent << 23));
  int ex = dexponent + n - multiplier;
  dexponent = ex;
  uint val = sign | (ex << 23) | (dval_ui & 0x007fffff);
  int ex1 = dexponent + multiplier;
  ex1 = -ex1 + 25;
  dval_ui = (((dval_ui & 0x007fffff) | 0x800000) >> ex1);
  dval_ui = dexponent > 0 ? val : dval_ui;
  dval_ui = dexponent > 254 ? 0x7f800000 : dval_ui; /*overflow*/
  dval_ui = dexponent < -multiplier ? 0 : dval_ui;  /*underflow*/
  dval_ui = dval_ui | sign;
  val_f = __clc_as_float(dval_ui);

  exponent += n;

  val = sign | (exponent << 23) | (val_ui & 0x007fffff);
  ex1 = exponent + multiplier;
  ex1 = -ex1 + 25;
  val_ui = (((val_ui & 0x007fffff) | 0x800000) >> ex1);
  val_ui = exponent > 0 ? val : val_ui;
  val_ui = exponent > 254 ? 0x7f800000 : val_ui; /*overflow*/
  val_ui = exponent < -multiplier ? 0 : val_ui;  /*underflow*/
  val_ui = val_ui | sign;

  val_ui = dexp == 0 ? dval_ui : val_ui;
  val_f = __clc_as_float(val_ui);

  val_f = (__clc_isnan(x) || __clc_isinf(x) || val_x == 0) ? x : val_f;
  return val_f;
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __clc_ldexp(double x, int n) {
  long l = __clc_as_ulong(x);
  int e = (l >> 52) & 0x7ff;
  long s = l & 0x8000000000000000;

  ulong ux = __clc_as_ulong(x * 0x1.0p+53);
  int de = ((int)(ux >> 52) & 0x7ff) - 53;
  int c = e == 0;
  e = c ? de : e;

  ux = c ? ux : l;

  int v = e + n;
  v = __clc_clamp(v, -0x7ff, 0x7ff);

  ux &= ~EXPBITS_DP64;

  double mr = __clc_as_double(ux | ((ulong)(v + 53) << 52));
  mr = mr * 0x1.0p-53;

  mr = v > 0 ? __clc_as_double(ux | ((ulong)v << 52)) : mr;

  mr = v == 0x7ff ? __clc_as_double(s | PINFBITPATT_DP64) : mr;
  mr = v < -53 ? __clc_as_double(s) : mr;

  mr = ((n == 0) | __clc_isinf(x) | (x == 0)) ? x : mr;
  return mr;
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_ldexp(half x, int n) {
  return (half)__clc_ldexp((float)x, n);
}

#endif

#define __CLC_FUNCTION __clc_ldexp
#define __CLC_ARG2_TYPE int
#define __CLC_BODY <clc/shared/binary_def_scalarize.inc>
#include <clc/math/gentype.inc>
