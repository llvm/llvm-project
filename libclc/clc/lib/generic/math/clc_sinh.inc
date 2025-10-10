//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sinh(__CLC_GENTYPE x) {
  // After dealing with special cases the computation is split into regions as
  // follows. abs(x) >= max_sinh_arg: sinh(x) = sign(x)*Inf abs(x) >=
  // small_threshold: sinh(x) = sign(x)*exp(abs(x))/2 computed using the
  // splitexp and scaleDouble functions as for exp_amd(). abs(x) <
  // small_threshold: compute p = exp(y) - 1 and then z = 0.5*(p+(p/(p+1.0)))
  // sinh(x) is then sign(x)*z.

  const __CLC_GENTYPE max_sinh_arg = 0x1.65a9fap+6f;
  const __CLC_GENTYPE small_threshold = 0x1.0a2b24p+3f;

  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_GENTYPE y = __clc_fabs(x);
  __CLC_UINTN aux = __CLC_AS_UINTN(y);
  __CLC_UINTN xs = ux ^ aux;

  // We find the integer part y0 of y and the increment dy = y - y0. We then
  // compute z = sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy) where sinh(y0)
  // and cosh(y0) are tabulated above.
  __CLC_INTN ind = __CLC_CONVERT_INTN(y);
  ind = __CLC_CONVERT_UINTN(ind) > 36U ? 0 : ind;

  __CLC_GENTYPE dy = y - __CLC_CONVERT_GENTYPE(ind);
  __CLC_GENTYPE dy2 = dy * dy;

  __CLC_GENTYPE sdy = __clc_mad(
      dy2,
      __clc_mad(
          dy2,
          __clc_mad(
              dy2,
              __clc_mad(
                  dy2,
                  __clc_mad(dy2,
                            __clc_mad(dy2, 0.7746188980094184251527126e-12f,
                                      0.160576793121939886190847e-9f),
                            0.250521176994133472333666e-7f),
                  0.275573191913636406057211e-5f),
              0.198412698413242405162014e-3f),
          0.833333333333329931873097e-2f),
      0.166666666666666667013899e0f);
  sdy = __clc_mad(sdy, dy * dy2, dy);

  __CLC_GENTYPE cdy = __clc_mad(
      dy2,
      __clc_mad(
          dy2,
          __clc_mad(
              dy2,
              __clc_mad(
                  dy2,
                  __clc_mad(dy2,
                            __clc_mad(dy2, 0.1163921388172173692062032e-10f,
                                      0.208744349831471353536305e-8f),
                            0.275573350756016588011357e-6f),
                  0.248015872460622433115785e-4f),
              0.138888888889814854814536e-2f),
          0.416666666666660876512776e-1f),
      0.500000000000000005911074e0f);
  cdy = __clc_mad(cdy, dy2, 1.0f);

  __CLC_GENTYPE sinhcoshh = __CLC_USE_TABLE(sinhcosh_tbl_head, ind);
  __CLC_GENTYPE sinhcosht = __CLC_USE_TABLE(sinhcosh_tbl_tail, ind);
  __CLC_GENTYPE z = __clc_mad(sinhcosht, sdy, sinhcoshh * cdy);
  z = __CLC_AS_GENTYPE(xs | __CLC_AS_UINTN(z));

  // When y is large enough so that the negative exponential is negligible,
  // so sinh(y) is approximated by sign(x)*exp(y)/2.
  __CLC_GENTYPE t = __clc_exp(y - 0x1.62e500p-1f);
  __CLC_GENTYPE zsmall = __clc_mad(0x1.a0210ep-18f, t, t);
  zsmall = __CLC_AS_GENTYPE(xs | __CLC_AS_UINTN(zsmall));
  z = y >= small_threshold ? zsmall : z;

  // Corner cases
  __CLC_GENTYPE zinf = __CLC_AS_GENTYPE(PINFBITPATT_SP32 | xs);
  z = y >= max_sinh_arg ? zinf : z;
  z = aux > PINFBITPATT_SP32 || aux < 0x38800000U ? x : z;

  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sinh(__CLC_GENTYPE x) {
  // After dealing with special cases the computation is split into
  // regions as follows:
  //
  // abs(x) >= max_sinh_arg:
  // sinh(x) = sign(x)*Inf
  //
  // abs(x) >= small_threshold:
  // sinh(x) = sign(x)*exp(abs(x))/2 computed using the
  // splitexp and scaleDouble functions as for exp_amd().
  //
  // abs(x) < small_threshold:
  // compute p = exp(y) - 1 and then z = 0.5*(p+(p/(p+1.0)))
  // sinh(x) is then sign(x)*z.

  // 0x408633ce8fb9f87e
  const __CLC_GENTYPE max_sinh_arg = 7.10475860073943977113e+02;

  // This is where exp(-x) is insignificant compared to exp(x) = ln(2^27)
  const __CLC_GENTYPE small_threshold = 0x1.2b708872320e2p+4;

  __CLC_GENTYPE y = __clc_fabs(x);

  // In this range we find the integer part y0 of y
  // and the increment dy = y - y0. We then compute
  // z = sinh(y) = sinh(y0)cosh(dy) + cosh(y0)sinh(dy)
  // where sinh(y0) and cosh(y0) are obtained from tables

  __CLC_INTN ind = __clc_min(__CLC_CONVERT_INTN(y), 36);
  __CLC_GENTYPE dy = y - __CLC_CONVERT_GENTYPE(ind);
  __CLC_GENTYPE dy2 = dy * dy;

  __CLC_GENTYPE sdy =
      dy * dy2 *
      __clc_fma(
          dy2,
          __clc_fma(
              dy2,
              __clc_fma(
                  dy2,
                  __clc_fma(
                      dy2,
                      __clc_fma(dy2,
                                __clc_fma(dy2, 0.7746188980094184251527126e-12,
                                          0.160576793121939886190847e-9),
                                0.250521176994133472333666e-7),
                      0.275573191913636406057211e-5),
                  0.198412698413242405162014e-3),
              0.833333333333329931873097e-2),
          0.166666666666666667013899e0);

  __CLC_GENTYPE cdy =
      dy2 *
      __clc_fma(
          dy2,
          __clc_fma(
              dy2,
              __clc_fma(
                  dy2,
                  __clc_fma(
                      dy2,
                      __clc_fma(dy2,
                                __clc_fma(dy2, 0.1163921388172173692062032e-10,
                                          0.208744349831471353536305e-8),
                                0.275573350756016588011357e-6),
                      0.248015872460622433115785e-4),
                  0.138888888889814854814536e-2),
              0.416666666666660876512776e-1),
          0.500000000000000005911074e0);

  // At this point sinh(dy) is approximated by dy + sdy.
  // Shift some significant bits from dy to sdy.
  __CLC_GENTYPE sdy1 =
      __CLC_AS_GENTYPE(__CLC_AS_ULONGN(dy) & 0xfffffffff8000000UL);
  __CLC_GENTYPE sdy2 = sdy + (dy - sdy1);

  __CLC_GENTYPE cl = __CLC_USE_TABLE(cosh_tbl_head, ind);
  __CLC_GENTYPE ct = __CLC_USE_TABLE(cosh_tbl_tail, ind);
  __CLC_GENTYPE sl = __CLC_USE_TABLE(sinh_tbl_head, ind);
  __CLC_GENTYPE st = __CLC_USE_TABLE(sinh_tbl_tail, ind);

  __CLC_GENTYPE z =
      __clc_fma(cl, sdy1,
                __clc_fma(sl, cdy,
                          __clc_fma(cl, sdy2,
                                    __clc_fma(ct, sdy1,
                                              __clc_fma(st, cdy, ct * sdy2)) +
                                        st))) +
      sl;

  // Other cases
  z = (y < 0x1.0p-28) || __clc_isnan(x) || __clc_isinf(x) ? y : z;

  __CLC_GENTYPE t = __clc_exp(y - 0x1.62e42fefa3800p-1);
  t = __clc_fma(t, -0x1.ef35793c76641p-45, t);
  z = y >= small_threshold ? t : z;
  z = y >= max_sinh_arg ? __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64) : z;

  return __clc_copysign(z, x);
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_sinh(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_sinh(__CLC_CONVERT_FLOATN(x)));
}

#endif
