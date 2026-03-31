//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanh(__CLC_GENTYPE x) {
  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_UINTN ax = ux & EXSIGNBIT_SP32;
  __CLC_UINTN xs = ux ^ ax;

  // |x| > 1 or NaN
  __CLC_GENTYPE z = __CLC_GENTYPE_NAN;

  // |x| == 1
  __CLC_GENTYPE t = __CLC_AS_GENTYPE(xs | PINFBITPATT_SP32);
  z = ax == 0x3f800000U ? t : z;

  // 1/2 <= |x| < 1
  t = __CLC_AS_GENTYPE(ax);
  t = MATH_DIVIDE(2.0f * t, 1.0f - t);
  t = 0.5f * __clc_log1p(t);
  t = __CLC_AS_GENTYPE(xs | __CLC_AS_UINTN(t));
  z = ax < 0x3f800000U ? t : z;

  // |x| < 1/2
  t = x * x;
  __CLC_GENTYPE a = __clc_mad(
      __clc_mad(0.92834212715e-2f, t, -0.28120347286e0f), t, 0.39453629046e0f);
  __CLC_GENTYPE b = __clc_mad(__clc_mad(0.45281890445e0f, t, -0.15537744551e1f),
                              t, 0.11836088638e1f);
  __CLC_GENTYPE p = MATH_DIVIDE(a, b);
  t = __clc_mad(x * t, p, x);
  z = ax < 0x3f000000 ? t : z;

  // |x| < 2^-13
  z = ax < 0x39000000U ? x : z;

  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanh(__CLC_GENTYPE x) {
  __CLC_GENTYPE absx = __clc_fabs(x);

  __CLC_GENTYPE ret = absx == 1.0
                          ? __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64)
                          : __CLC_GENTYPE_NAN;

  // |x| >= 0.5
  // Note that atanh(x) = 0.5 * ln((1+x)/(1-x))
  // For greater accuracy we use
  // ln((1+x)/(1-x)) = ln(1 + 2x/(1-x)) = log1p(2x/(1-x)).
  __CLC_GENTYPE r = 0.5 * __clc_log1p(2.0 * absx / (1.0 - absx));
  ret = absx < 1.0 ? r : ret;

  r = -ret;
  ret = x < 0.0 ? r : ret;

  // Arguments up to 0.5 in magnitude are
  // approximated by a [5,5] minimax polynomial
  __CLC_GENTYPE t = x * x;

  __CLC_GENTYPE pn = __clc_fma(
      t,
      __clc_fma(t,
                __clc_fma(t,
                          __clc_fma(t,
                                    __clc_fma(t, -0.10468158892753136958e-3,
                                              0.28728638600548514553e-1),
                                    -0.28180210961780814148e0),
                          0.88468142536501647470e0),
                -0.11028356797846341457e1),
      0.47482573589747356373e0);

  __CLC_GENTYPE pd = __clc_fma(
      t,
      __clc_fma(t,
                __clc_fma(t,
                          __clc_fma(t,
                                    __clc_fma(t, -0.35861554370169537512e-1,
                                              0.49561196555503101989e0),
                                    -0.22608883748988489342e1),
                          0.45414700626084508355e1),
                -0.41631933639693546274e1),
      0.14244772076924206909e1);

  r = __clc_fma(x * t, pn / pd, x);
  ret = absx < 0.5 ? r : ret;

  return ret;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_atanh(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_atanh(__CLC_CONVERT_FLOATN(x)));
}

#endif
