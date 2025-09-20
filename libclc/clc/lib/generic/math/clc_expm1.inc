//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/* Refer to the exp routine for the underlying algorithm */
#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_expm1(__CLC_GENTYPE x) {
  // 128*log2 : 88.722839111673
  const __CLC_GENTYPE X_MAX = 0x1.62e42ep+6f;
  // -149*log2 : -103.27892990343184
  const __CLC_GENTYPE X_MIN = -0x1.9d1da0p+6f;
  // 64/log2 : 92.332482616893657
  const __CLC_GENTYPE R_64_BY_LOG2 = 0x1.715476p+6f;
  // log2/64 lead: 0.0108032227
  const __CLC_GENTYPE R_LOG2_BY_64_LD = 0x1.620000p-7f;
  // log2/64 tail: 0.0000272020388
  const __CLC_GENTYPE R_LOG2_BY_64_TL = 0x1.c85fdep-16f;

  __CLC_UINTN xi = __CLC_AS_UINTN(x);
  __CLC_INTN n = __CLC_CONVERT_INTN(x * R_64_BY_LOG2);
  __CLC_GENTYPE fn = __CLC_CONVERT_GENTYPE(n);

  __CLC_INTN j = n & 0x3f;
  __CLC_INTN m = n >> 6;

  __CLC_GENTYPE r =
      __clc_mad(fn, -R_LOG2_BY_64_TL, __clc_mad(fn, -R_LOG2_BY_64_LD, x));

  // Truncated Taylor series
  __CLC_GENTYPE z2 = __clc_mad(
      r * r, __clc_mad(r, __clc_mad(r, 0x1.555556p-5f, 0x1.555556p-3f), 0.5f),
      r);

  __CLC_GENTYPE m2 = __CLC_AS_GENTYPE((m + EXPBIAS_SP32) << EXPSHIFTBITS_SP32);
  __CLC_GENTYPE exp_head = __CLC_USE_TABLE(exp_tbl_ep_head, j);
  __CLC_GENTYPE exp_tail = __CLC_USE_TABLE(exp_tbl_ep_tail, j);

  __CLC_GENTYPE two_to_jby64_h = exp_head * m2;
  __CLC_GENTYPE two_to_jby64_t = exp_tail * m2;
  __CLC_GENTYPE two_to_jby64 = two_to_jby64_h + two_to_jby64_t;

  z2 = __clc_mad(z2, two_to_jby64, two_to_jby64_t) + (two_to_jby64_h - 1.0f);
  // Make subnormals work
  z2 = x == 0.f ? x : z2;
  z2 = x < X_MIN || m < -24 ? -1.0f : z2;
  z2 = x > X_MAX ? __CLC_AS_GENTYPE((__CLC_UINTN)PINFBITPATT_SP32) : z2;
  z2 = __clc_isnan(x) ? x : z2;

  return z2;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_expm1(__CLC_GENTYPE x) {
  const __CLC_GENTYPE max_expm1_arg = 709.8;
  const __CLC_GENTYPE min_expm1_arg = -37.42994775023704;
  // 0x3FCC8FF7C79A9A22 = log(1+1/4)
  const __CLC_GENTYPE log_OnePlus_OneByFour = 0.22314355131420976;
  // 0xBFD269621134DB93 = log(1-1/4)
  const __CLC_GENTYPE log_OneMinus_OneByFour = -0.28768207245178096;
  const __CLC_GENTYPE sixtyfour_by_lnof2 =
      92.33248261689366; // 0x40571547652b82fe
  const __CLC_GENTYPE lnof2_by_64_head =
      0.010830424696223417; // 0x3f862e42fefa0000
  const __CLC_GENTYPE lnof2_by_64_tail =
      2.5728046223276688e-14; // 0x3d1cf79abc9e3b39

  // First, assume log(1-1/4) < x < log(1+1/4) i.e  -0.28768 < x < 0.22314
  __CLC_GENTYPE u = __CLC_AS_GENTYPE(__CLC_AS_ULONGN(x) & 0xffffffffff000000UL);
  __CLC_GENTYPE v = x - u;
  __CLC_GENTYPE y = u * u * 0.5;
  __CLC_GENTYPE z = v * (x + u) * 0.5;

  __CLC_GENTYPE q = __clc_fma(
      x,
      __clc_fma(
          x,
          __clc_fma(
              x,
              __clc_fma(
                  x,
                  __clc_fma(
                      x,
                      __clc_fma(x,
                                __clc_fma(x,
                                          __clc_fma(x, 2.4360682937111612e-8,
                                                    2.7582184028154370e-7),
                                          2.7558212415361945e-6),
                                2.4801576918453420e-5),
                      1.9841269447671544e-4),
                  1.3888888890687830e-3),
              8.3333333334012270e-3),
          4.1666666666665560e-2),
      1.6666666666666632e-1);
  q *= x * x * x;

  __CLC_GENTYPE z1g = (u + y) + (q + (v + z));
  __CLC_GENTYPE z1 = x + (y + (q + z));
  z1 = y >= 0x1.0p-7 ? z1g : z1;

  // Now assume outside interval around 0
  __CLC_INTN n = __CLC_CONVERT_INTN(x * sixtyfour_by_lnof2);
  __CLC_INTN j = n & 0x3f;
  __CLC_INTN m = n >> 6;

  __CLC_GENTYPE f1 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_head, j);
  __CLC_GENTYPE f2 = __CLC_USE_TABLE(two_to_jby64_ep_tbl_tail, j);
  __CLC_GENTYPE f = f1 + f2;

  __CLC_GENTYPE dn = __CLC_CONVERT_GENTYPE(-n);
  __CLC_GENTYPE r =
      __clc_fma(dn, lnof2_by_64_tail, __clc_fma(dn, lnof2_by_64_head, x));

  q = __clc_fma(r,
                __clc_fma(r,
                          __clc_fma(r,
                                    __clc_fma(r, 1.38889490863777199667e-03,
                                              8.33336798434219616221e-03),
                                    4.16666666662260795726e-02),
                          1.66666666665260878863e-01),
                5.00000000000000008883e-01);
  q = __clc_fma(r * r, q, r);

  __CLC_GENTYPE twopm = __CLC_AS_GENTYPE(__CLC_CONVERT_LONGN(m + EXPBIAS_DP64)
                                         << EXPSHIFTBITS_DP64);
  __CLC_GENTYPE twopmm = __CLC_AS_GENTYPE(__CLC_CONVERT_LONGN(EXPBIAS_DP64 - m)
                                          << EXPSHIFTBITS_DP64);

  // Computations for m > 52, including where result is close to Inf
  __CLC_ULONGN uval = __CLC_AS_ULONGN(0x1.0p+1023 * (f1 + (f * q + (f2))));
  __CLC_INTN e = __CLC_CONVERT_INTN(uval >> EXPSHIFTBITS_DP64) + 1;

  __CLC_GENTYPE zme1024 = __CLC_AS_GENTYPE(
      (__CLC_CONVERT_ULONGN(e) << EXPSHIFTBITS_DP64) | (uval & MANTBITS_DP64));
  zme1024 = __CLC_CONVERT_LONGN(e == 2047)
                ? __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64)
                : zme1024;

  __CLC_GENTYPE zmg52 = twopm * (f1 + __clc_fma(f, q, f2 - twopmm));
  zmg52 = __CLC_CONVERT_LONGN(m == 1024) ? zme1024 : zmg52;

  // For m < 53
  __CLC_GENTYPE zml53 =
      twopm * ((f1 - twopmm) + __clc_fma(f1, q, f2 * (1.0 + q)));

  // For m < -7
  __CLC_GENTYPE zmln7 = __clc_fma(twopm, f1 + __clc_fma(f, q, f2), -1.0);

  z = __CLC_CONVERT_LONGN(m < 53) ? zml53 : zmg52;
  z = __CLC_CONVERT_LONGN(m < -7) ? zmln7 : z;
  z = x > log_OneMinus_OneByFour && x < log_OnePlus_OneByFour ? z1 : z;
  z = x > max_expm1_arg ? __CLC_AS_GENTYPE((__CLC_ULONGN)PINFBITPATT_DP64) : z;
  z = x < min_expm1_arg ? -1.0 : z;

  return z;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_expm1(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_expm1(__CLC_CONVERT_FLOATN(x)));
}

#endif
