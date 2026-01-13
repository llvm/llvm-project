//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Algorithm:
//
//  e^x = 2^(x/ln(2)) = 2^(x*(64/ln(2))/64)
//
//  x*(64/ln(2)) = n + f, |f| <= 0.5, n is integer
//  n = 64*m + j,   0 <= j < 64
//
//  e^x = 2^((64*m + j + f)/64)
//      = (2^m) * (2^(j/64)) * 2^(f/64)
//      = (2^m) * (2^(j/64)) * e^(f*(ln(2)/64))
//
//  f = x*(64/ln(2)) - n
//  r = f*(ln(2)/64) = x - n*(ln(2)/64)
//
//  e^x = (2^m) * (2^(j/64)) * e^r
//
//  (2^(j/64)) is precomputed
//
//  e^r = 1 + r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//  e^r = 1 + q
//
//  q = r + (r^2)/2! + (r^3)/3! + (r^4)/4! + (r^5)/5!
//
//  e^x = (2^m) * ( (2^(j/64)) + q*(2^(j/64)) )
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_exp10(__CLC_GENTYPE x) {
  // 128*log2/log10 : 38.53183944498959
  const __CLC_GENTYPE X_MAX = 0x1.344134p+5f;
  // -149*log2/log10 : -44.8534693539332
  const __CLC_GENTYPE X_MIN = -0x1.66d3e8p+5f;
  // 64*log10/log2 : 212.6033980727912
  const __CLC_GENTYPE R_64_BY_LOG10_2 = 0x1.a934f0p+7f;
  // log2/(64 * log10) lead : 0.004699707
  const __CLC_GENTYPE R_LOG10_2_BY_64_LD = 0x1.340000p-8f;
  // log2/(64 * log10) tail : 0.00000388665057
  const __CLC_GENTYPE R_LOG10_2_BY_64_TL = 0x1.04d426p-18f;
  const __CLC_GENTYPE R_LN10 = 0x1.26bb1cp+1f;

  __CLC_INTN return_nan = __clc_isnan(x);
  __CLC_INTN return_inf = x > X_MAX;
  __CLC_INTN return_zero = x < X_MIN;

  __CLC_INTN n = __CLC_CONVERT_INTN(x * R_64_BY_LOG10_2);

  __CLC_GENTYPE fn = __CLC_CONVERT_GENTYPE(n);
  __CLC_INTN j = n & 0x3f;
  __CLC_INTN m = n >> 6;
  __CLC_INTN m2 = m << EXPSHIFTBITS_SP32;
  __CLC_GENTYPE r;

  r = R_LN10 *
      __clc_mad(fn, -R_LOG10_2_BY_64_TL, __clc_mad(fn, -R_LOG10_2_BY_64_LD, x));

  // Truncated Taylor series for e^r
  __CLC_GENTYPE z2 =
      __clc_mad(__clc_mad(__clc_mad(r, 0x1.555556p-5f, 0x1.555556p-3f), r,
                          0x1.000000p-1f),
                r * r, r);

  __CLC_GENTYPE two_to_jby64 = __CLC_USE_TABLE(exp_tbl, j);
  z2 = __clc_mad(two_to_jby64, z2, two_to_jby64);

  __CLC_GENTYPE z2s = z2 * __CLC_AS_GENTYPE((__CLC_UINTN)0x1 << (m + 149));
  __CLC_GENTYPE z2n = __CLC_AS_GENTYPE(__CLC_AS_INTN(z2) + m2);
  z2 = m <= -126 ? z2s : z2n;

  z2 = return_inf ? __CLC_AS_GENTYPE((__CLC_UINTN)PINFBITPATT_SP32) : z2;
  z2 = return_zero ? 0.0f : z2;
  z2 = return_nan ? x : z2;
  return z2;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD __CLC_GENTYPE __clc_exp10(__CLC_GENTYPE x) {
  // 1024*ln(2)/ln(10)
  const __CLC_GENTYPE X_MAX = 0x1.34413509f79ffp+8;
  // -1074*ln(2)/ln(10)
  const __CLC_GENTYPE X_MIN = -0x1.434e6420f4374p+8;
  // 64*ln(10)/ln(2)
  const __CLC_GENTYPE R_64_BY_LOG10_2 = 0x1.a934f0979a371p+7;
  // head ln(2)/(64*ln(10))
  const __CLC_GENTYPE R_LOG10_2_BY_64_LD = 0x1.3441350000000p-8;
  // tail ln(2)/(64*ln(10))
  const __CLC_GENTYPE R_LOG10_2_BY_64_TL = 0x1.3ef3fde623e25p-37;
  // ln(10)
  const __CLC_GENTYPE R_LN10 = 0x1.26bb1bbb55516p+1;

  __CLC_INTN n = __CLC_CONVERT_INTN(x * R_64_BY_LOG10_2);

  __CLC_GENTYPE dn = __CLC_CONVERT_GENTYPE(n);

  __CLC_INTN j = n & 0x3f;
  __CLC_INTN m = n >> 6;

  __CLC_GENTYPE r = R_LN10 * __clc_fma(-R_LOG10_2_BY_64_TL, dn,
                                       __clc_fma(-R_LOG10_2_BY_64_LD, dn, x));

  // 6 term tail of Taylor expansion of e^r
  __CLC_GENTYPE z2 =
      r * __clc_fma(
              r,
              __clc_fma(r,
                        __clc_fma(r,
                                  __clc_fma(r,
                                            __clc_fma(r, 0x1.6c16c16c16c17p-10,
                                                      0x1.1111111111111p-7),
                                            0x1.5555555555555p-5),
                                  0x1.5555555555555p-3),
                        0x1.0000000000000p-1),
              1.0);

  __CLC_GENTYPE tv0 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_head, j);
  __CLC_GENTYPE tv1 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_tail, j);
  z2 = __clc_fma(tv0 + tv1, z2, tv1) + tv0;

  __CLC_INTN small_value =
      (m < -1022) || ((m == -1022) && __CLC_CONVERT_INTN(z2 < 1.0));

  __CLC_INTN n1 = m >> 2;
  __CLC_INTN n2 = m - n1;
  __CLC_GENTYPE z3 =
      z2 * __CLC_AS_GENTYPE((__CLC_CONVERT_LONGN(n1) + 1023) << 52);
  z3 *= __CLC_AS_GENTYPE((__CLC_CONVERT_LONGN(n2) + 1023) << 52);

  z2 = __clc_ldexp(z2, m);
  z2 = __CLC_CONVERT_LONGN(small_value) ? z3 : z2;

  z2 = __clc_isnan(x) ? x : z2;

  z2 = x > X_MAX ? __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64) : z2;
  z2 = x < X_MIN ? 0.0 : z2;

  return z2;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_exp10(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_exp10(__CLC_CONVERT_FLOATN(x)));
}

#endif
