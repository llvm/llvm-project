//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes arcsin(x).
//
// The incoming argument is first reduced by noting that arcsin(x) is invalid
// for abs(x) > 1 and arcsin(-x) = -arcsin(x).
//
// For denormal and small arguments, arcsin(x) = x to machine accuracy.
//
// Remaining argument ranges are handled as follows:
// * For abs(x) <= 0.5 use:
//     arcsin(x) = x + x^3 * R(x^2)
//   where R(x^2) is a rational minimax approximation to (arcsin(x) - x)/x^3.
// * For abs(x) > 0.5 exploit the identity:
//     arcsin(x) = pi/2 - 2 * arcsin(sqrt(1 - x)/2)
//   together with the above rational approximation, and reconstruct the terms
//   carefully.
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_asin(__CLC_GENTYPE x) {
  // 0x33a22168
  const __CLC_GENTYPE piby2_tail = __CLC_FP_LIT(7.5497894159e-08);
  // 0x3f490fda
  const __CLC_GENTYPE hpiby2_head = __CLC_FP_LIT(7.8539812565e-01);
  // 0x3fc90fdb
  const __CLC_GENTYPE piby2 = __CLC_FP_LIT(1.5707963705e+00);

  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_UINTN aux = ux & EXSIGNBIT_SP32;
  __CLC_UINTN xs = ux ^ aux;
  __CLC_GENTYPE spiby2 = __CLC_AS_GENTYPE(xs | __CLC_AS_UINTN(piby2));
  __CLC_INTN xexp = __CLC_AS_INTN(aux >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;
  __CLC_GENTYPE y = __CLC_AS_GENTYPE(aux);

  // abs(x) >= 0.5
  __CLC_INTN transform = xexp >= -1;

  __CLC_GENTYPE y2 = y * y;
  __CLC_GENTYPE rt = __CLC_FP_LIT(0.5) * (__CLC_FP_LIT(1.0) - y);
  __CLC_GENTYPE r = transform ? rt : y2;

  // Use a rational approximation for [0.0, 0.5]
  __CLC_GENTYPE a =
      __clc_mad(r,
                __clc_mad(r,
                          __clc_mad(r, -0.00396137437848476485201154797087F,
                                    -0.0133819288943925804214011424456F),
                          -0.0565298683201845211985026327361F),
                0.184161606965100694821398249421F);

  __CLC_GENTYPE b = __clc_mad(r, -0.836411276854206731913362287293F,
                              1.10496961524520294485512696706F);
  __CLC_GENTYPE u = r * MATH_DIVIDE(a, b);

  __CLC_GENTYPE s = __clc_sqrt(r);
  __CLC_GENTYPE s1 = __CLC_AS_GENTYPE(__CLC_AS_UINTN(s) & 0xffff0000);
  __CLC_GENTYPE c = MATH_DIVIDE(__clc_mad(-s1, s1, r), s + s1);
  __CLC_GENTYPE p = __clc_mad(2.0f * s, u, -__clc_mad(c, -2.0f, piby2_tail));
  __CLC_GENTYPE q = __clc_mad(s1, -2.0f, hpiby2_head);
  __CLC_GENTYPE vt = hpiby2_head - (p - q);
  __CLC_GENTYPE v = __clc_mad(y, u, y);
  v = transform ? vt : v;

  __CLC_GENTYPE ret = __CLC_AS_GENTYPE(xs | __CLC_AS_UINTN(v));
  ret = aux > 0x3f800000U ? __CLC_GENTYPE_NAN : ret;
  ret = aux == 0x3f800000U ? spiby2 : ret;
  ret = xexp < -14 ? x : ret;

  return ret;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_asin(__CLC_GENTYPE x) {
  // 0x3c91a62633145c07
  const __CLC_GENTYPE piby2_tail = __CLC_FP_LIT(6.1232339957367660e-17);
  // 0x3fe921fb54442d18
  const __CLC_GENTYPE hpiby2_head = 7.8539816339744831e-01;
  // 0x3ff921fb54442d18
  const __CLC_GENTYPE piby2 = 1.5707963267948965e+00;

  __CLC_GENTYPE y = __clc_fabs(x);
  __CLC_LONGN xneg = x < __CLC_FP_LIT(0.0);
  __CLC_INTN xexp = __CLC_CONVERT_INTN(
      (__CLC_AS_ULONGN(y) >> EXPSHIFTBITS_DP64) - EXPBIAS_DP64);

  // abs(x) >= 0.5
  __CLC_LONGN transform = __CLC_CONVERT_LONGN(xexp >= -1);

  __CLC_GENTYPE rt = __CLC_FP_LIT(0.5) * (__CLC_FP_LIT(1.0) - y);
  __CLC_GENTYPE y2 = y * y;
  __CLC_GENTYPE r = transform ? rt : y2;

  // Use a rational approximation for [0.0, 0.5]

  __CLC_GENTYPE un = __clc_fma(
      r,
      __clc_fma(
          r,
          __clc_fma(r,
                    __clc_fma(r,
                              __clc_fma(r, 0.0000482901920344786991880522822991,
                                        0.00109242697235074662306043804220),
                              -0.0549989809235685841612020091328),
                    0.275558175256937652532686256258),
          -0.445017216867635649900123110649),
      0.227485835556935010735943483075);

  __CLC_GENTYPE ud = __clc_fma(
      r,
      __clc_fma(r,
                __clc_fma(r,
                          __clc_fma(r, 0.105869422087204370341222318533,
                                    -0.943639137032492685763471240072),
                          2.76568859157270989520376345954),
                -3.28431505720958658909889444194),
      1.36491501334161032038194214209);

  __CLC_GENTYPE u = r * MATH_DIVIDE(un, ud);

  // Reconstruct asin carefully in transformed region
  __CLC_GENTYPE s = __clc_sqrt(r);
  __CLC_GENTYPE sh =
      __CLC_AS_GENTYPE(__CLC_AS_ULONGN(s) & 0xffffffff00000000UL);
  __CLC_GENTYPE c = MATH_DIVIDE(__clc_fma(-sh, sh, r), s + sh);
  __CLC_GENTYPE p = __clc_fma(2.0 * s, u, -__clc_fma(-2.0, c, piby2_tail));
  __CLC_GENTYPE q = __clc_fma(-2.0, sh, hpiby2_head);
  __CLC_GENTYPE vt = hpiby2_head - (p - q);
  __CLC_GENTYPE v = __clc_fma(y, u, y);
  v = transform ? vt : v;

  v = __CLC_CONVERT_LONGN(xexp < -28) ? y : v;
  v = __CLC_CONVERT_LONGN(xexp >= 0) ? __CLC_GENTYPE_NAN : v;
  v = y == 1.0 ? piby2 : v;

  return xneg ? -v : v;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_asin(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_asin(__CLC_CONVERT_FLOATN(x)));
}

#endif
