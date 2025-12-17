//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_cbrt(__CLC_GENTYPE x) {
  __CLC_UINTN xi = __CLC_AS_UINTN(x);
  __CLC_UINTN axi = __CLC_AS_UINTN(__clc_fabs(x));
  __CLC_UINTN xsign = axi ^ xi;
  xi = axi;

  __CLC_INTN m = __CLC_AS_INTN(xi >> EXPSHIFTBITS_SP32) - EXPBIAS_SP32;

  // Treat subnormals
  __CLC_UINTN xisub = __CLC_AS_UINTN(__CLC_AS_GENTYPE(xi | 0x3f800000) - 1.0f);
  __CLC_INTN msub = __CLC_AS_INTN(xisub >> EXPSHIFTBITS_SP32) - 253;
  __CLC_INTN c = m == -127;
  xi = c ? xisub : xi;
  m = c ? msub : m;

  __CLC_INTN m3 = m / 3;
  __CLC_INTN rem = m - m3 * 3;
  __CLC_GENTYPE mf = __CLC_AS_GENTYPE((m3 + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);

  __CLC_UINTN indx = (xi & 0x007f0000) + ((xi & 0x00008000) << 1);
  __CLC_GENTYPE f = __CLC_AS_GENTYPE((xi & MANTBITS_SP32) | 0x3f000000) -
                    __CLC_AS_GENTYPE(indx | 0x3f000000);

  indx >>= 16;
  __CLC_GENTYPE r = f * __CLC_USE_TABLE(log_inv_tbl, __CLC_AS_INTN(indx));
  __CLC_GENTYPE poly = __clc_mad(__clc_mad(r, 0x1.f9add4p-5f, -0x1.c71c72p-4f),
                                 r * r, r * 0x1.555556p-2f);

  // This could also be done with a 5-element table
  __CLC_GENTYPE remH = 0x1.428000p-1f;
  __CLC_GENTYPE remT = 0x1.45f31ap-14f;

  remH = rem == -1 ? 0x1.964000p-1f : remH;
  remT = rem == -1 ? 0x1.fea53ep-13f : remT;

  remH = rem == 0 ? 0x1.000000p+0f : remH;
  remT = rem == 0 ? 0x0.000000p+0f : remT;

  remH = rem == 1 ? 0x1.428000p+0f : remH;
  remT = rem == 1 ? 0x1.45f31ap-13f : remT;

  remH = rem == 2 ? 0x1.964000p+0f : remH;
  remT = rem == 2 ? 0x1.fea53ep-12f : remT;

  __CLC_GENTYPE cbrtH = __CLC_USE_TABLE(cbrt_tbl_head, __CLC_AS_INTN(indx));
  __CLC_GENTYPE cbrtT = __CLC_USE_TABLE(cbrt_tbl_tail, __CLC_AS_INTN(indx));

  __CLC_GENTYPE bH = cbrtH * remH;
  __CLC_GENTYPE bT =
      __clc_mad(cbrtH, remT, __clc_mad(cbrtT, remH, cbrtT * remT));

  __CLC_GENTYPE z = __clc_mad(poly, bH, __clc_mad(poly, bT, bT)) + bH;
  z *= mf;
  z = __CLC_AS_GENTYPE(__CLC_AS_UINTN(z) | xsign);
  c = axi >= EXPBITS_SP32 || axi == 0;
  z = c ? x : z;
  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_cbrt(__CLC_GENTYPE x) {
  __CLC_LONGN return_x = __clc_isinf(x) || __clc_isnan(x) || x == 0.0;
  __CLC_ULONGN ux = __CLC_AS_ULONGN(__clc_fabs(x));
  __CLC_INTN m = __CLC_CONVERT_INTN(ux >> EXPSHIFTBITS_DP64) - 1023;

  // Treat subnormals
  __CLC_ULONGN uxs =
      __CLC_AS_ULONGN(__CLC_AS_GENTYPE(0x3ff0000000000000UL | ux) - 1.0);
  __CLC_INTN ms = m + __CLC_CONVERT_INTN(uxs >> EXPSHIFTBITS_DP64) - 1022;

  __CLC_INTN c = m == -1023;
  ux = __CLC_CONVERT_LONGN(c) ? uxs : ux;
  m = c ? ms : m;

  __CLC_INTN mby3 = m / 3;
  __CLC_INTN rem = m - 3 * mby3;

  __CLC_GENTYPE mf = __CLC_AS_GENTYPE(__CLC_CONVERT_ULONGN(mby3 + 1023) << 52);

  ux &= 0x000fffffffffffffUL;
  __CLC_GENTYPE Y = __CLC_AS_GENTYPE(0x3fe0000000000000UL | ux);

  // nearest integer
  __CLC_INTN index = __CLC_CONVERT_INTN(ux >> 43);
  index = (0x100 | (index >> 1)) + (index & 1);
  __CLC_GENTYPE F = __CLC_CONVERT_GENTYPE(index) * 0x1.0p-9;

  __CLC_GENTYPE f = Y - F;
  __CLC_GENTYPE r = f * __CLC_USE_TABLE(cbrt_inv_tbl, index - 256);

  __CLC_GENTYPE z =
      r * __clc_fma(
              r,
              __clc_fma(r,
                        __clc_fma(r,
                                  __clc_fma(r,
                                            __clc_fma(r, -0x1.8090d6221a247p-6,
                                                      0x1.ee7113506ac13p-6),
                                            -0x1.511e8d2b3183bp-5),
                                  0x1.f9add3c0ca458p-5),
                        -0x1.c71c71c71c71cp-4),
              0x1.5555555555555p-2);

  __CLC_GENTYPE Rem_h = __CLC_USE_TABLE(cbrt_rem_tbl_head, rem + 2);
  __CLC_GENTYPE Rem_t = __CLC_USE_TABLE(cbrt_rem_tbl_tail, rem + 2);

  __CLC_GENTYPE F_h = __CLC_USE_TABLE(cbrt_dbl_tbl_head, index - 256);
  __CLC_GENTYPE F_t = __CLC_USE_TABLE(cbrt_dbl_tbl_tail, index - 256);

  __CLC_GENTYPE b_h = F_h * Rem_h;
  __CLC_GENTYPE b_t = __clc_fma(Rem_t, F_h, __clc_fma(F_t, Rem_h, F_t * Rem_t));

  __CLC_GENTYPE ans = __clc_fma(z, b_h, __clc_fma(z, b_t, b_t)) + b_h;
  ans = __clc_copysign(ans * mf, x);
  return return_x ? x : ans;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_cbrt(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_cbrt(__CLC_CONVERT_FLOATN(x)));
}

#endif
