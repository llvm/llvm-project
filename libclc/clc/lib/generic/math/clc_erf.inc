//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunPro, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 */

/* double erf(double x)
 * double erfc(double x)
 *                             x
 *                      2      |\
 *     erf(x)  =  ---------  | exp(-t*t)dt
 *                    sqrt(pi) \|
 *                             0
 *
 *     erfc(x) =  1-erf(x)
 *  Note that
 *                erf(-x) = -erf(x)
 *                erfc(-x) = 2 - erfc(x)
 *
 * Method:
 *        1. For |x| in [0, 0.84375]
 *            erf(x)  = x + x*R(x^2)
 *          erfc(x) = 1 - erf(x)           if x in [-.84375,0.25]
 *                  = 0.5 + ((0.5-x)-x*R)  if x in [0.25,0.84375]
 *           where R = P/Q where P is an odd poly of degree 8 and
 *           Q is an odd poly of degree 10.
 *                                                 -57.90
 *                        | R - (erf(x)-x)/x | <= 2
 *
 *
 *           Remark. The formula is derived by noting
 *          erf(x) = (2/sqrt(pi))*(x - x^3/3 + x^5/10 - x^7/42 + ....)
 *           and that
 *          2/sqrt(pi) = 1.128379167095512573896158903121545171688
 *           is close to one. The interval is chosen because the fix
 *           point of erf(x) is near 0.6174 (i.e., erf(x)=x when x is
 *           near 0.6174), and by some experiment, 0.84375 is chosen to
 *            guarantee the error is less than one ulp for erf.
 *
 *      2. For |x| in [0.84375,1.25], let s = |x| - 1, and
 *         c = 0.84506291151 rounded to single (24 bits)
 *                 erf(x)  = sign(x) * (c  + P1(s)/Q1(s))
 *                 erfc(x) = (1-c)  - P1(s)/Q1(s) if x > 0
 *                          1+(c+P1(s)/Q1(s))    if x < 0
 *                 |P1/Q1 - (erf(|x|)-c)| <= 2**-59.06
 *           Remark: here we use the taylor series expansion at x=1.
 *                erf(1+s) = erf(1) + s*Poly(s)
 *                         = 0.845.. + P1(s)/Q1(s)
 *           That is, we use rational approximation to approximate
 *                        erf(1+s) - (c = (single)0.84506291151)
 *           Note that |P1/Q1|< 0.078 for x in [0.84375,1.25]
 *           where
 *                P1(s) = degree 6 poly in s
 *                Q1(s) = degree 6 poly in s
 *
 *      3. For x in [1.25,1/0.35(~2.857143)],
 *                 erfc(x) = (1/x)*exp(-x*x-0.5625+R1/S1)
 *                 erf(x)  = 1 - erfc(x)
 *           where
 *                R1(z) = degree 7 poly in z, (z=1/x^2)
 *                S1(z) = degree 8 poly in z
 *
 *      4. For x in [1/0.35,28]
 *                 erfc(x) = (1/x)*exp(-x*x-0.5625+R2/S2) if x > 0
 *                        = 2.0 - (1/x)*exp(-x*x-0.5625+R2/S2) if -6<x<0
 *                        = 2.0 - tiny                (if x <= -6)
 *                 erf(x)  = sign(x)*(1.0 - erfc(x)) if x < 6, else
 *                 erf(x)  = sign(x)*(1.0 - tiny)
 *           where
 *                R2(z) = degree 6 poly in z, (z=1/x^2)
 *                S2(z) = degree 7 poly in z
 *
 *      Note1:
 *           To compute exp(-x*x-0.5625+R/S), let s be a single
 *           precision number and s := x; then
 *                -x*x = -s*s + (s-x)*(s+x)
 *                exp(-x*x-0.5626+R/S) =
 *                        exp(-s*s-0.5625)*exp((s-x)*(s+x)+R/S);
 *      Note2:
 *           Here 4 and 5 make use of the asymptotic series
 *                          exp(-x*x)
 *                erfc(x) ~ ---------- * ( 1 + Poly(1/x^2) )
 *                          x*sqrt(pi)
 *           We use rational approximation to approximate
 *              g(s)=f(1/x^2) = log(erfc(x)*x) - x*x + 0.5625
 *           Here is the error bound for R1/S1 and R2/S2
 *              |R1/S1 - f(x)|  < 2**(-62.57)
 *              |R2/S2 - f(x)|  < 2**(-61.52)
 *
 *      5. For inf > x >= 28
 *                 erf(x)  = sign(x) *(1 - tiny)  (raise inexact)
 *                 erfc(x) = tiny*tiny (raise underflow) if x > 0
 *                        = 2 - tiny if x<0
 *
 *      7. Special case:
 *                 erf(0)  = 0, erf(inf)  = 1, erf(-inf) = -1,
 *                 erfc(0) = 1, erfc(inf) = 0, erfc(-inf) = 2,
 *                   erfc/erf(NaN) is NaN
 */

#pragma OPENCL FP_CONTRACT OFF

#if __CLC_FPSIZE == 32

static _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_erf_lt_1(__CLC_FLOATN ax) {
  __CLC_FLOATN t = ax * ax;

  __CLC_FLOATN u0 = __clc_mad(t, -0x1.268bc2p-11f, 0x1.420828p-8f);
  __CLC_FLOATN u1 = __clc_mad(t, u0, -0x1.b5937p-6f);
  __CLC_FLOATN u2 = __clc_mad(t, u1, 0x1.ce077cp-4f);
  __CLC_FLOATN u3 = __clc_mad(t, u2, -0x1.81266p-2f);
  __CLC_FLOATN p = __clc_mad(t, u3, 0x1.06eba0p-3f);

  return __clc_fma(ax, p, ax);
}

static _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_erf_ge_1(__CLC_FLOATN ax) {
  __CLC_FLOATN u0 = __clc_mad(ax, 0x1.1d3156p-16f, -0x1.8d129p-12f);
  __CLC_FLOATN u1 = __clc_mad(ax, u0, 0x1.f9a6d2p-9f);
  __CLC_FLOATN u2 = __clc_mad(ax, u1, -0x1.8c3164p-6f);
  __CLC_FLOATN u3 = __clc_mad(ax, u2, 0x1.b4e9c8p-4f);
  __CLC_FLOATN u4 = __clc_mad(ax, u3, 0x1.4515fap-1f);
  __CLC_FLOATN p = __clc_mad(ax, u4, 0x1.078e50p-3f);

  return 1.0f - __clc_exp(-__clc_fma(ax, p, ax));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_erf(__CLC_FLOATN x) {
  __CLC_FLOATN ax = __clc_fabs(x);
  __CLC_FLOATN ret = ax < 1.0f ? __clc_erf_lt_1(ax) : __clc_erf_ge_1(ax);
  return __clc_copysign(ret, x);
}

#elif __CLC_FPSIZE == 64

static _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_erf_lt_1(__CLC_DOUBLEN ax) {
  __CLC_DOUBLEN t = ax * ax;

  __CLC_DOUBLEN u0 =
      __clc_mad(t, -0x1.ab15c51d2ebebp-31, 0x1.d6e3ddfeb1f49p-27);
  __CLC_DOUBLEN u1 = __clc_mad(t, u0, -0x1.5bfe76384472p-23);
  __CLC_DOUBLEN u2 = __clc_mad(t, u1, 0x1.b97e44280cfb9p-20);
  __CLC_DOUBLEN u3 = __clc_mad(t, u2, -0x1.f4ca204c771c5p-17);
  __CLC_DOUBLEN u4 = __clc_mad(t, u3, 0x1.f9a2b75531772p-14);
  __CLC_DOUBLEN u5 = __clc_mad(t, u4, -0x1.c02db0149d904p-11);
  __CLC_DOUBLEN u6 = __clc_mad(t, u5, 0x1.565bccf7e2856p-8);
  __CLC_DOUBLEN u7 = __clc_mad(t, u6, -0x1.b82ce311ee09bp-6);
  __CLC_DOUBLEN u8 = __clc_mad(t, u7, 0x1.ce2f21a0408d1p-4);
  __CLC_DOUBLEN u9 = __clc_mad(t, u8, -0x1.812746b0379b2p-2);
  __CLC_DOUBLEN p = __clc_mad(t, u9, 0x1.06eba8214db68p-3);

  return __clc_mad(ax, p, ax);
}

static _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_erf_ge_1(__CLC_DOUBLEN ax) {
  __CLC_DOUBLEN t0 =
      __clc_mad(ax, 0x1.98d37c14b24bep-58, -0x1.145a3502a41cdp-51);
  __CLC_DOUBLEN t1 = __clc_mad(ax, t0, 0x1.62deed735f9ecp-46);
  __CLC_DOUBLEN t2 = __clc_mad(ax, t1, -0x1.1ffe55552ca22p-41);
  __CLC_DOUBLEN t3 = __clc_mad(ax, t2, 0x1.4b9ba7074b644p-37);
  __CLC_DOUBLEN t4 = __clc_mad(ax, t3, -0x1.20345a78ce24p-33);
  __CLC_DOUBLEN t5 = __clc_mad(ax, t4, 0x1.88b7a0cefddd8p-30);
  __CLC_DOUBLEN t6 = __clc_mad(ax, t5, -0x1.aded48c94b617p-27);
  __CLC_DOUBLEN t7 = __clc_mad(ax, t6, 0x1.803aa312306dp-24);
  __CLC_DOUBLEN t8 = __clc_mad(ax, t7, -0x1.1b0106f4c5a9bp-21);
  __CLC_DOUBLEN t9 = __clc_mad(ax, t8, 0x1.58c0e7cfd79aep-19);
  __CLC_DOUBLEN t10 = __clc_mad(ax, t9, -0x1.59e386410fdf7p-17);
  __CLC_DOUBLEN t11 = __clc_mad(ax, t10, 0x1.192fc1f9b1786p-15);
  __CLC_DOUBLEN t12 = __clc_mad(ax, t11, -0x1.62cf3f4634b2ep-14);
  __CLC_DOUBLEN t13 = __clc_mad(ax, t12, 0x1.314dfb42f7e4bp-13);
  __CLC_DOUBLEN t14 = __clc_mad(ax, t13, -0x1.2cb68c047288ap-14);
  __CLC_DOUBLEN t15 = __clc_mad(ax, t14, -0x1.038ff7bbcce25p-11);
  __CLC_DOUBLEN t16 = __clc_mad(ax, t15, 0x1.a9466ae1babaep-10);
  __CLC_DOUBLEN t17 = __clc_mad(ax, t16, -0x1.58be1e65a6063p-13);
  __CLC_DOUBLEN t18 = __clc_mad(ax, t17, -0x1.39bc16738ee3ap-6);
  __CLC_DOUBLEN t19 = __clc_mad(ax, t18, 0x1.a4fbc28146b69p-4);
  __CLC_DOUBLEN t20 = __clc_mad(ax, t19, 0x1.45f2da69750c4p-1);
  __CLC_DOUBLEN p = __clc_mad(ax, t20, 0x1.06ebb919fcca8p-3);

  return 1.0 - __clc_exp(-__clc_mad(ax, p, ax));
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_erf(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN ax = __clc_fabs(x);
  __CLC_DOUBLEN ret = ax < 1.0 ? __clc_erf_lt_1(ax) : __clc_erf_ge_1(ax);
  return __clc_copysign(ret, x);
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_HALFN __clc_erf(__CLC_HALFN x) {
  return __CLC_CONVERT_HALFN(__clc_erf(__CLC_CONVERT_FLOATN(x)));
}

#endif
