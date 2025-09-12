//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Computes natural log(x). Algorithm based on:
// Ping-Tak Peter Tang
// "Table-driven implementation of the logarithm function in IEEE floating-point
// arithmetic"
// ACM Transactions on Mathematical Software (TOMS) Volume 16, Issue 4 (December
// 1990)
//
//===----------------------------------------------------------------------===//

#if __CLC_FPSIZE == 64

#define LN0 8.33333333333317923934e-02
#define LN1 1.25000000037717509602e-02
#define LN2 2.23213998791944806202e-03
#define LN3 4.34887777707614552256e-04

#define LF0 8.33333333333333593622e-02
#define LF1 1.24999999978138668903e-02
#define LF2 2.23219810758559851206e-03

_CLC_DEF _CLC_OVERLOAD void __clc_ep_log(__CLC_GENTYPE x,
                                         private __CLC_INTN *xexp,
                                         private __CLC_GENTYPE *r1,
                                         private __CLC_GENTYPE *r2) {
  __CLC_LONGN near_one = x >= 0x1.e0faap-1 && x <= 0x1.1082cp+0;

  __CLC_ULONGN ux = __CLC_AS_ULONGN(x);
  __CLC_ULONGN uxs =
      __CLC_AS_ULONGN(__CLC_AS_GENTYPE(0x03d0000000000000UL | ux) - 0x1.0p-962);
  __CLC_LONGN c = ux < IMPBIT_DP64;
  ux = c ? uxs : ux;
  __CLC_INTN expadjust =
      __CLC_CONVERT_INTN(c ? (__CLC_LONGN)60 : (__CLC_LONGN)0);

  // Store the exponent of x in xexp and put f into the range [0.5,1)
  __CLC_INTN xexp1 = __CLC_CONVERT_INTN((ux >> EXPSHIFTBITS_DP64) & 0x7ff) -
                     EXPBIAS_DP64 - expadjust;
  __CLC_GENTYPE f = __CLC_AS_GENTYPE(HALFEXPBITS_DP64 | (ux & MANTBITS_DP64));
  *xexp = __CLC_CONVERT_INTN(near_one) ? 0 : xexp1;

  __CLC_GENTYPE r = x - 1.0;
  __CLC_GENTYPE u1 = MATH_DIVIDE(r, 2.0 + r);
  __CLC_GENTYPE ru1 = -r * u1;
  u1 = u1 + u1;

  __CLC_INTN index = __CLC_CONVERT_INTN(ux >> 45);
  index = ((0x80 | (index & 0x7e)) >> 1) + (index & 0x1);

  __CLC_GENTYPE f1 = __CLC_CONVERT_GENTYPE(index) * 0x1.0p-7;
  __CLC_GENTYPE f2 = f - f1;
  __CLC_GENTYPE u2 = MATH_DIVIDE(f2, __clc_fma(0.5, f2, f1));

  __CLC_GENTYPE z1 = __CLC_USE_TABLE(ln_tbl_lo, (index - 64));
  __CLC_GENTYPE q = __CLC_USE_TABLE(ln_tbl_hi, (index - 64));

  z1 = near_one ? r : z1;
  q = near_one ? 0.0 : q;
  __CLC_GENTYPE u = near_one ? u1 : u2;
  __CLC_GENTYPE v = u * u;

  __CLC_GENTYPE cc = near_one ? ru1 : u2;

  __CLC_GENTYPE z21 =
      __clc_fma(v, __clc_fma(v, __clc_fma(v, LN3, LN2), LN1), LN0);
  __CLC_GENTYPE z22 = __clc_fma(v, __clc_fma(v, LF2, LF1), LF0);
  __CLC_GENTYPE z2 = near_one ? z21 : z22;
  z2 = __clc_fma(u * v, z2, cc) + q;

  *r1 = z1;
  *r2 = z2;
}

#endif
