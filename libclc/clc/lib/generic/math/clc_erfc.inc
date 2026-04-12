//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// Some of this implementation is based on ideas from Sun libm.
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
/*
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

// Computes exp(x**2) * erfc(x)
static _CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN
__clc_erfc_scaled(__CLC_FLOATN x) {
  __CLC_FLOATN n = x - 2.0f;
  __CLC_FLOATN d = x + 2.0f;
  __CLC_FLOATN r = __clc_recip_fast(d);
  __CLC_FLOATN q = n * r;
  __CLC_FLOATN e = __clc_fma(-q, x, __clc_fma(q + 1.0f, -2.0f, x));
  q = __clc_fma(r, e, q);

  // Polynomial in q
  __CLC_FLOATN u0 = __clc_mad(q, -0x1.adf188p-12f, -0x1.45aea6p-10f);
  __CLC_FLOATN u1 = __clc_mad(q, u0, 0x1.5a5f68p-10f);
  __CLC_FLOATN u2 = __clc_mad(q, u1, 0x1.1b44cep-7f);
  __CLC_FLOATN u3 = __clc_mad(q, u2, -0x1.082b62p-7f);
  __CLC_FLOATN u4 = __clc_mad(q, u3, -0x1.bc143p-5f);
  __CLC_FLOATN u5 = __clc_mad(q, u4, 0x1.4ffc54p-3f);
  __CLC_FLOATN u6 = __clc_mad(q, u5, -0x1.5407fap-3f);
  __CLC_FLOATN u7 = __clc_mad(q, u6, -0x1.7bf616p-4f);
  __CLC_FLOATN p = __clc_mad(q, u7, 0x1.1ba038p-2f);

  // Final evaluation
  __CLC_FLOATN tx = x + x;
  d = 1.0f + tx;
  r = __clc_recip_fast(d);
  q = __clc_fma(p, r, r);
  e = __clc_fma(-q, tx, 1.0f) + (p - q);

  return __clc_fma(r, e, q);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_FLOATN __clc_erfc(__CLC_FLOATN x) {
  __CLC_FLOATN ax = __clc_fabs(x);
  __CLC_FLOATN x2h = -x * x;
  __CLC_FLOATN x2l = __clc_fma(-x, x, -x2h);
  __CLC_FLOATN e = __clc_exp(x2h);
  e = __clc_fma(e, x2l, e);
  __CLC_FLOATN ret = e * __clc_erfc_scaled(ax);
  ret = ax > 0x1.41bbf8p+3f ? 0.0f : ret;
  __CLC_FLOATN nret = 2.0f - ret;
  return x < 0.0f ? nret : ret;
}

#elif __CLC_FPSIZE == 64

// Computes exp(x**2) * erfc(x)
static _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN
__clc_erfc_scaled(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN n = x - 4.0;
  __CLC_DOUBLEN d = x + 4.0;
  __CLC_DOUBLEN r = __clc_recip_fast(d);
  __CLC_DOUBLEN q = n * r;
  __CLC_DOUBLEN e = __clc_mad(-q, x, __clc_mad(q + 1.0, -4.0, x));
  q = __clc_fma(r, e, q);

  __CLC_DOUBLEN u0 =
      __clc_mad(q, -0x1.1f39d54df3c0ep-27, -0x1.1166337cfa789p-27);
  __CLC_DOUBLEN u1 = __clc_mad(q, u0, 0x1.b45f1d9802b82p-24);
  __CLC_DOUBLEN u2 = __clc_mad(q, u1, 0x1.d90488a03dcdbp-25);
  __CLC_DOUBLEN u3 = __clc_mad(q, u2, -0x1.b87b02eba62d8p-21);
  __CLC_DOUBLEN u4 = __clc_mad(q, u3, 0x1.5104ba56e15f1p-22);
  __CLC_DOUBLEN u5 = __clc_mad(q, u4, 0x1.7f29f71c907dep-18);
  __CLC_DOUBLEN u6 = __clc_mad(q, u5, -0x1.78f5c2cd770fbp-17);
  __CLC_DOUBLEN u7 = __clc_mad(q, u6, -0x1.995fb76d0a51ap-16);
  __CLC_DOUBLEN u8 = __clc_mad(q, u7, 0x1.3be2ec022d0edp-13);
  __CLC_DOUBLEN u9 = __clc_mad(q, u8, -0x1.a1deb2fdbf62ep-13);
  __CLC_DOUBLEN u10 = __clc_mad(q, u9, -0x1.8d4ac3689fc43p-11);
  __CLC_DOUBLEN u11 = __clc_mad(q, u10, 0x1.49c67192d909bp-8);
  __CLC_DOUBLEN u12 = __clc_mad(q, u11, -0x1.09623852ff07p-6);
  __CLC_DOUBLEN u13 = __clc_mad(q, u12, 0x1.3079edfadea8fp-5);
  __CLC_DOUBLEN u14 = __clc_mad(q, u13, -0x1.0fb06dff6591p-4);
  __CLC_DOUBLEN u15 = __clc_mad(q, u14, 0x1.7fee004de8f32p-4);
  __CLC_DOUBLEN u16 = __clc_mad(q, u15, -0x1.9ddb23c3dbeb3p-4);
  __CLC_DOUBLEN u17 = __clc_mad(q, u16, 0x1.16ecefcfa693p-4);
  __CLC_DOUBLEN u18 = __clc_mad(q, u17, 0x1.f7f5df66fb8a3p-7);
  __CLC_DOUBLEN u19 = __clc_mad(q, u18, -0x1.1df1ad154a2a8p-3);
  __CLC_DOUBLEN p = __clc_mad(q, u19, 0x1.dd2c8b74febf8p-3);

  __CLC_DOUBLEN tx = x + x;
  d = 1.0 + tx;
  r = __clc_recip_fast(d);
  q = __clc_mad(p, r, r);
  e = __clc_mad(-q, tx, 1.0) + (p - q);

  return __clc_mad(r, e, q);
}

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_DOUBLEN __clc_erfc(__CLC_DOUBLEN x) {
  __CLC_DOUBLEN ax = __clc_fabs(x);
  __CLC_DOUBLEN x2h = -x * x;
  __CLC_DOUBLEN x2l = __clc_mad(-x, x, -x2h);
  __CLC_DOUBLEN e = __clc_exp(x2h);
  e = __clc_mad(e, x2l, e);
  __CLC_DOUBLEN ret = e * __clc_erfc_scaled(ax);
  ret = ax > 0x1.b39dc41e48bfcp+4 ? 0.0f : ret;
  __CLC_DOUBLEN nret = 2.0 - ret;
  return x < 0.0 ? nret : ret;
}

#elif __CLC_FPSIZE == 16

_CLC_OVERLOAD _CLC_CONST _CLC_DEF __CLC_HALFN __clc_erfc(__CLC_HALFN x) {
  return __CLC_CONVERT_HALFN(__clc_erfc(__CLC_CONVERT_FLOATN(x)));
}

#endif
