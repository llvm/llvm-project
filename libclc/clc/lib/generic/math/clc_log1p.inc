//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes natural log(1+x). Algorithm based on:
// Ping-Tak Peter Tang
// "Table-driven implementation of the logarithm function in IEEE floating-point
// arithmetic" ACM Transactions on Mathematical Software (TOMS) Volume 16, Issue
// 4 (December 1990)
//
// Note that we use a lookup table of size 64 rather than 128, and compensate by
// having extra terms in the minimax polynomial for the kernel approximation.
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 32

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_log1p(__CLC_GENTYPE x) {
  __CLC_GENTYPE w = x;
  __CLC_UINTN ux = __CLC_AS_UINTN(x);
  __CLC_UINTN ax = ux & EXSIGNBIT_SP32;

  // |x| < 2^-4
  __CLC_GENTYPE u2 = MATH_DIVIDE(x, 2.0f + x);
  __CLC_GENTYPE u = u2 + u2;
  __CLC_GENTYPE v = u * u;
  // 2/(5 * 2^5), 2/(3 * 2^3)
  __CLC_GENTYPE zsmall =
      __clc_mad(-u2, x, __clc_mad(v, 0x1.99999ap-7f, 0x1.555556p-4f) * v * u) +
      x;

  // |x| >= 2^-4
  ux = __CLC_AS_UINTN(x + 1.0f);

  __CLC_INTN m = __CLC_AS_INTN((ux >> EXPSHIFTBITS_SP32) & 0xff) - EXPBIAS_SP32;
  __CLC_GENTYPE mf = __CLC_CONVERT_GENTYPE(m);
  __CLC_UINTN indx = (ux & 0x007f0000) + ((ux & 0x00008000) << 1);
  __CLC_GENTYPE F = __CLC_AS_GENTYPE(indx | 0x3f000000);

  // x > 2^24
  __CLC_GENTYPE fg24 = F - __CLC_AS_GENTYPE(0x3f000000 | (ux & MANTBITS_SP32));

  // x <= 2^24
  __CLC_UINTN xhi = ux & 0xffff8000;
  __CLC_GENTYPE xh = __CLC_AS_GENTYPE(xhi);
  __CLC_GENTYPE xt = (1.0f - xh) + w;
  __CLC_UINTN xnm = ((~(xhi & 0x7f800000)) - 0x00800000) & 0x7f800000;
  xt = xt * __CLC_AS_GENTYPE(xnm) * 0.5f;
  __CLC_GENTYPE fl24 =
      F - __CLC_AS_GENTYPE(0x3f000000 | (xhi & MANTBITS_SP32)) - xt;

  __CLC_GENTYPE f = mf > 24.0f ? fg24 : fl24;

  indx = indx >> 16;
  __CLC_GENTYPE r = f * __CLC_USE_TABLE(log_inv_tbl, __CLC_CONVERT_INTN(indx));

  // 1/3, 1/2
  __CLC_GENTYPE poly =
      __clc_mad(__clc_mad(r, 0x1.555556p-2f, 0x1.0p-1f), r * r, r);

  const __CLC_GENTYPE LOG2_HEAD = 0x1.62e000p-1f;  // 0.693115234
  const __CLC_GENTYPE LOG2_TAIL = 0x1.0bfbe8p-15f; // 0.0000319461833

  __CLC_GENTYPE tv0 = __CLC_USE_TABLE(loge_tbl_lo, __CLC_AS_INTN(indx));
  __CLC_GENTYPE tv1 = __CLC_USE_TABLE(loge_tbl_hi, __CLC_AS_INTN(indx));
  __CLC_GENTYPE z1 = __clc_mad(mf, LOG2_HEAD, tv0);
  __CLC_GENTYPE z2 = __clc_mad(mf, LOG2_TAIL, -poly) + tv1;
  __CLC_GENTYPE z = z1 + z2;

  z = ax < 0x3d800000U ? zsmall : z;

  // Edge cases
  z = ax >= PINFBITPATT_SP32 ? w : z;
  z = w < -1.0f ? __CLC_GENTYPE_NAN : z;
  z = w == -1.0f ? __CLC_AS_GENTYPE((__CLC_UINTN)NINFBITPATT_SP32) : z;
  // Fix subnormals
  z = ax < 0x33800000 ? x : z;

  return z;
}

#elif __CLC_FPSIZE == 64

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_log1p(__CLC_GENTYPE x) {
  // Process Inside the threshold now
  __CLC_ULONGN ux = __CLC_AS_ULONGN((__CLC_GENTYPE)1.0 + x);
  __CLC_INTN xexp =
      __CLC_CONVERT_INTN((ux >> EXPSHIFTBITS_DP64) & 0x7ff) - EXPBIAS_DP64;
  __CLC_GENTYPE f =
      __CLC_AS_GENTYPE((__CLC_ULONGN)ONEEXPBITS_DP64 | (ux & MANTBITS_DP64));

  __CLC_INTN j = __CLC_CONVERT_INTN(ux >> 45);
  j = ((0x80 | (j & 0x7e)) >> 1) + (j & 0x1);
  __CLC_GENTYPE f1 = __CLC_CONVERT_GENTYPE(j) * 0x1.0p-6;
  j -= 64;

  __CLC_GENTYPE f2temp = f - f1;
  __CLC_GENTYPE m2 =
      __CLC_AS_GENTYPE(__CLC_CONVERT_ULONGN(0x3ff - xexp) << EXPSHIFTBITS_DP64);
  __CLC_GENTYPE f2l = __clc_fma(m2, x, m2 - f1);
  __CLC_GENTYPE f2g = __clc_fma(m2, x, -f1) + m2;
  __CLC_GENTYPE f2 =
      __CLC_CONVERT_LONGN(xexp <= MANTLENGTH_DP64 - 1) ? f2l : f2g;
  f2 = __CLC_CONVERT_LONGN(xexp <= -2 || (xexp >= MANTLENGTH_DP64 + 8)) ? f2temp
                                                                        : f2;

  __CLC_GENTYPE z1 = __CLC_USE_TABLE(ln_tbl_lo, j);
  __CLC_GENTYPE q = __CLC_USE_TABLE(ln_tbl_hi, j);

  __CLC_GENTYPE u = MATH_DIVIDE(f2, __clc_fma(0.5, f2, f1));
  __CLC_GENTYPE v = u * u;

  __CLC_GENTYPE poly = v * __clc_fma(v,
                                     __clc_fma(v, 2.23219810758559851206e-03,
                                               1.24999999978138668903e-02),
                                     8.33333333333333593622e-02);

  // log2_lead and log2_tail sum to an extra-precise version of log(2)
  // 0x3fe62e42e0000000
  const __CLC_GENTYPE log2_lead = 6.93147122859954833984e-01;
  // 0x3e6efa39ef35793c
  const __CLC_GENTYPE log2_tail = 5.76999904754328540596e-08;

  __CLC_GENTYPE z2 = q + __clc_fma(u, poly, u);
  __CLC_GENTYPE dxexp = __CLC_CONVERT_GENTYPE(xexp);
  __CLC_GENTYPE r1 = __clc_fma(dxexp, log2_lead, z1);
  __CLC_GENTYPE r2 = __clc_fma(dxexp, log2_tail, z2);
  __CLC_GENTYPE result1 = r1 + r2;

  // Process Outside the threshold now
  __CLC_GENTYPE r = x;
  u = r / (2.0 + r);
  __CLC_GENTYPE correction = r * u;
  u = u + u;
  v = u * u;
  r1 = r;

  poly = __clc_fma(v,
                   __clc_fma(v,
                             __clc_fma(v, 4.34887777707614552256e-04,
                                       2.23213998791944806202e-03),
                             1.25000000037717509602e-02),
                   8.33333333333317923934e-02);

  r2 = __clc_fma(u * v, poly, -correction);

  // The values exp(-1/16)-1 and exp(1/16)-1
  const __CLC_GENTYPE log1p_thresh1 = -0x1.f0540438fd5c3p-5;
  const __CLC_GENTYPE log1p_thresh2 = 0x1.082b577d34ed8p-4;
  __CLC_GENTYPE result2 = r1 + r2;
  result2 = x < log1p_thresh1 || x > log1p_thresh2 ? result1 : result2;

  result2 = __clc_isinf(x) ? x : result2;
  result2 = x < -1.0 ? __CLC_GENTYPE_NAN : result2;
  result2 =
      x == -1.0 ? __CLC_AS_GENTYPE((__CLC_ULONGN)NINFBITPATT_DP64) : result2;
  return result2;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_DEF __CLC_GENTYPE __clc_log1p(__CLC_GENTYPE x) {
  return __CLC_CONVERT_GENTYPE(__clc_log1p(__CLC_CONVERT_FLOATN(x)));
}

#endif
