/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "config.h"
#include "math.h"
#include <clc/clc.h>
#include <clc/clcmacro.h>
#include <clc/relational/clc_isinf.h>
#include <clc/relational/clc_isnan.h>
#include <clc/shared/clc_clamp.h>

_CLC_DEF _CLC_OVERLOAD float __clc_ldexp(float x, int n) {

  if (!__clc_fp32_subnormals_supported()) {

    // This treats subnormals as zeros
    int i = as_int(x);
    int e = (i >> 23) & 0xff;
    int m = i & 0x007fffff;
    int s = i & 0x80000000;
    int v = add_sat(e, n);
    v = __clc_clamp(v, 0, 0xff);
    int mr = e == 0 | v == 0 | v == 0xff ? 0 : m;
    int c = e == 0xff;
    mr = c ? m : mr;
    int er = c ? e : v;
    er = e ? er : e;
    return as_float(s | (er << 23) | mr);
  }

  /* supports denormal values */
  const int multiplier = 24;
  float val_f;
  uint val_ui;
  uint sign;
  int exponent;
  val_ui = as_uint(x);
  sign = val_ui & 0x80000000;
  val_ui = val_ui & 0x7fffffff; /* remove the sign bit */
  int val_x = val_ui;

  exponent = val_ui >> 23; /* get the exponent */
  int dexp = exponent;

  /* denormal support */
  int fbh =
      127 - (as_uint((float)(as_float(val_ui | 0x3f800000) - 1.0f)) >> 23);
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
  val_f = as_float(dval_ui);

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
  val_f = as_float(val_ui);

  val_f = __clc_isnan(x) | __clc_isinf(x) | val_x == 0 ? x : val_f;
  return val_f;
}

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

_CLC_DEF _CLC_OVERLOAD double __clc_ldexp(double x, int n) {
  long l = as_ulong(x);
  int e = (l >> 52) & 0x7ff;
  long s = l & 0x8000000000000000;

  ulong ux = as_ulong(x * 0x1.0p+53);
  int de = ((int)(ux >> 52) & 0x7ff) - 53;
  int c = e == 0;
  e = c ? de : e;

  ux = c ? ux : l;

  int v = e + n;
  v = __clc_clamp(v, -0x7ff, 0x7ff);

  ux &= ~EXPBITS_DP64;

  double mr = as_double(ux | ((ulong)(v + 53) << 52));
  mr = mr * 0x1.0p-53;

  mr = v > 0 ? as_double(ux | ((ulong)v << 52)) : mr;

  mr = v == 0x7ff ? as_double(s | PINFBITPATT_DP64) : mr;
  mr = v < -53 ? as_double(s) : mr;

  mr = ((n == 0) | __clc_isinf(x) | (x == 0)) ? x : mr;
  return mr;
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_ldexp(half x, int n) {
  return (half)__clc_ldexp((float)x, n);
}

_CLC_BINARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, __clc_ldexp, half, int);

#endif
