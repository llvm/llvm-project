//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This lgamma routine began with Sun's lgamma code from netlib.
// Their original copyright notice follows.

/* @(#)e_lgamma_r.c 1.3 95/01/18 */
/*
 * ====================================================
 * Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
 *
 * Developed at SunSoft, a Sun Microsystems, Inc. business.
 * Permission to use, copy, modify, and distribute this
 * software is freely granted, provided that this notice
 * is preserved.
 * ====================================================
 *
 */

/* __ieee754_lgamma_r(x, signgamp)
 * Reentrant version of the logarithm of the Gamma function
 * with user provide pointer for the sign of Gamma(x).
 *
 * Method:
 *   1. Argument Reduction for 0 < x <= 8
 *      Since gamma(1+s)=s*gamma(s), for x in [0,8], we may
 *      reduce x to a number in [1.5,2.5] by
 *              lgamma(1+s) = log(s) + lgamma(s)
 *      for example,
 *              lgamma(7.3) = log(6.3) + lgamma(6.3)
 *                          = log(6.3*5.3) + lgamma(5.3)
 *                          = log(6.3*5.3*4.3*3.3*2.3) + lgamma(2.3)
 *   2. Polynomial approximation of lgamma around its
 *      minimun ymin=1.461632144968362245 to maintain monotonicity.
 *      On [ymin-0.23, ymin+0.27] (i.e., [1.23164,1.73163]), use
 *              Let z = x-ymin;
 *              lgamma(x) = -1.214862905358496078218 + z^2*poly(z)
 *      where
 *              poly(z) is a 14 degree polynomial.
 *   2. Rational approximation in the primary interval [2,3]
 *      We use the following approximation:
 *              s = x-2.0;
 *              lgamma(x) = 0.5*s + s*P(s)/Q(s)
 *      with accuracy
 *              |P/Q - (lgamma(x)-0.5s)| < 2**-61.71
 *      Our algorithms are based on the following observation
 *
 *                             zeta(2)-1    2    zeta(3)-1    3
 * lgamma(2+s) = s*(1-Euler) + --------- * s  -  --------- * s  + ...
 *                                 2                 3
 *
 *      where Euler = 0.5771... is the Euler constant, which is very
 *      close to 0.5.
 *
 *   3. For x>=8, we have
 *      lgamma(x)~(x-0.5)log(x)-x+0.5*log(2pi)+1/(12x)-1/(360x**3)+....
 *      (better formula:
 *         lgamma(x)~(x-0.5)*(log(x)-1)-.5*(log(2pi)-1) + ...)
 *      Let z = 1/x, then we approximation
 *              f(z) = lgamma(x) - (x-0.5)(log(x)-1)
 *      by
 *                                  3       5             11
 *              w = w0 + w1*z + w2*z  + w3*z  + ... + w6*z
 *      where
 *              |w - f(z)| < 2**-58.74
 *
 *   4. For negative x, since (G is gamma function)
 *              -x*G(-x)*G(x) = pi/sin(pi*x),
 *      we have
 *              G(x) = pi/(sin(pi*x)*(-x)*G(-x))
 *      since G(-x) is positive, sign(G(x)) = sign(sin(pi*x)) for x<0
 *      Hence, for x<0, signgam = sign(sin(pi*x)) and
 *              lgamma(x) = log(|Gamma(x)|)
 *                        = log(pi/(|x*sin(pi*x)|)) - lgamma(-x);
 *      Note: one should avoid compute pi*(-x) directly in the
 *            computation of sin(pi*(-x)).
 *
 *   5. Special Cases
 *              lgamma(2+s) ~ s*(1-Euler) for tiny s
 *              lgamma(1)=lgamma(2)=0
 *              lgamma(x) ~ -log(x) for tiny x
 *              lgamma(0) = lgamma(inf) = inf
 *              lgamma(-integer) = +-inf
 *
 */

#ifdef __CLC_SCALAR

#if __CLC_FPSIZE == 32

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_LGAMMA_R_RET_GENTYPE
__clc_lgamma_r_stret(__CLC_FLOATN x) {
  const __CLC_FLOATN pi = 3.14159265358979311600e+00f;
  const __CLC_FLOATN a0 = 7.72156649015328655494e-02f;
  const __CLC_FLOATN a1 = 3.22467033424113591611e-01f;
  const __CLC_FLOATN a2 = 6.73523010531292681824e-02f;
  const __CLC_FLOATN a3 = 2.05808084325167332806e-02f;
  const __CLC_FLOATN a4 = 7.38555086081402883957e-03f;
  const __CLC_FLOATN a5 = 2.89051383673415629091e-03f;
  const __CLC_FLOATN a6 = 1.19270763183362067845e-03f;
  const __CLC_FLOATN a7 = 5.10069792153511336608e-04f;
  const __CLC_FLOATN a8 = 2.20862790713908385557e-04f;
  const __CLC_FLOATN a9 = 1.08011567247583939954e-04f;
  const __CLC_FLOATN a10 = 2.52144565451257326939e-05f;
  const __CLC_FLOATN a11 = 4.48640949618915160150e-05f;
  const __CLC_FLOATN tc = 1.46163214496836224576e+00f;
  const __CLC_FLOATN tf = -1.21486290535849611461e-01f;
  const __CLC_FLOATN tt = -3.63867699703950536541e-18f;
  const __CLC_FLOATN t0 = 4.83836122723810047042e-01f;
  const __CLC_FLOATN t1 = -1.47587722994593911752e-01f;
  const __CLC_FLOATN t2 = 6.46249402391333854778e-02f;
  const __CLC_FLOATN t3 = -3.27885410759859649565e-02f;
  const __CLC_FLOATN t4 = 1.79706750811820387126e-02f;
  const __CLC_FLOATN t5 = -1.03142241298341437450e-02f;
  const __CLC_FLOATN t6 = 6.10053870246291332635e-03f;
  const __CLC_FLOATN t7 = -3.68452016781138256760e-03f;
  const __CLC_FLOATN t8 = 2.25964780900612472250e-03f;
  const __CLC_FLOATN t9 = -1.40346469989232843813e-03f;
  const __CLC_FLOATN t10 = 8.81081882437654011382e-04f;
  const __CLC_FLOATN t11 = -5.38595305356740546715e-04f;
  const __CLC_FLOATN t12 = 3.15632070903625950361e-04f;
  const __CLC_FLOATN t13 = -3.12754168375120860518e-04f;
  const __CLC_FLOATN t14 = 3.35529192635519073543e-04f;
  const __CLC_FLOATN u0 = -7.72156649015328655494e-02f;
  const __CLC_FLOATN u1 = 6.32827064025093366517e-01f;
  const __CLC_FLOATN u2 = 1.45492250137234768737e+00f;
  const __CLC_FLOATN u3 = 9.77717527963372745603e-01f;
  const __CLC_FLOATN u4 = 2.28963728064692451092e-01f;
  const __CLC_FLOATN u5 = 1.33810918536787660377e-02f;
  const __CLC_FLOATN v1 = 2.45597793713041134822e+00f;
  const __CLC_FLOATN v2 = 2.12848976379893395361e+00f;
  const __CLC_FLOATN v3 = 7.69285150456672783825e-01f;
  const __CLC_FLOATN v4 = 1.04222645593369134254e-01f;
  const __CLC_FLOATN v5 = 3.21709242282423911810e-03f;
  const __CLC_FLOATN s0 = -7.72156649015328655494e-02f;
  const __CLC_FLOATN s1 = 2.14982415960608852501e-01f;
  const __CLC_FLOATN s2 = 3.25778796408930981787e-01f;
  const __CLC_FLOATN s3 = 1.46350472652464452805e-01f;
  const __CLC_FLOATN s4 = 2.66422703033638609560e-02f;
  const __CLC_FLOATN s5 = 1.84028451407337715652e-03f;
  const __CLC_FLOATN s6 = 3.19475326584100867617e-05f;
  const __CLC_FLOATN r1 = 1.39200533467621045958e+00f;
  const __CLC_FLOATN r2 = 7.21935547567138069525e-01f;
  const __CLC_FLOATN r3 = 1.71933865632803078993e-01f;
  const __CLC_FLOATN r4 = 1.86459191715652901344e-02f;
  const __CLC_FLOATN r5 = 7.77942496381893596434e-04f;
  const __CLC_FLOATN r6 = 7.32668430744625636189e-06f;
  const __CLC_FLOATN w0 = 4.18938533204672725052e-01f;
  const __CLC_FLOATN w1 = 8.33333333333329678849e-02f;
  const __CLC_FLOATN w2 = -2.77777777728775536470e-03f;
  const __CLC_FLOATN w3 = 7.93650558643019558500e-04f;
  const __CLC_FLOATN w4 = -5.95187557450339963135e-04f;
  const __CLC_FLOATN w5 = 8.36339918996282139126e-04f;
  const __CLC_FLOATN w6 = -1.63092934096575273989e-03f;
  const __CLC_FLOATN z1 = -0x1.2788d0p-1f;
  const __CLC_FLOATN z2 = 0x1.a51a66p-1f;
  const __CLC_FLOATN z3 = -0x1.9a4d56p-2f;
  const __CLC_FLOATN z4 = 0x1.151322p-2f;

  __CLC_FLOATN ax = __clc_fabs(x);
  __CLC_FLOATN ret;

  if (ax < 0x1.0p-6f) {
    ret = __clc_mad(ax,
                    __clc_mad(ax, __clc_mad(ax, __clc_mad(ax, z4, z3), z2), z1),
                    -__clc_log(ax));
  } else if (ax < 2.0f) {
    __CLC_INTN i;
    bool c;
    __CLC_FLOATN y, t;
    if (ax <= 0.9f) { // lgamma(x) = lgamma(x+1)-log(x)
      ret = -__clc_log(ax);
      y = 1.0f - ax;
      i = 0;

      c = ax < 0.7316f;
      t = ax - (tc - 1.0f);
      y = c ? t : y;
      i = c ? 1 : i;

      c = ax < 0.23164f;
      y = c ? ax : y;
      i = c ? 2 : i;
    } else {
      ret = 0.0f;
      y = 2.0f - ax;
      i = 0;

      c = ax < 1.7316f;
      t = ax - tc;
      y = c ? t : y;
      i = c ? 1 : y;

      c = ax < 1.23f;
      t = ax - 1.0f;
      y = c ? t : y;
      i = c ? 2 : i;
    }

    __CLC_FLOATN z, w, p1, p2, p3, p;
    switch (i) {
    case 0: {
      z = y * y;

      __CLC_FLOATN z2 = __clc_mad(z, a10, a8);
      __CLC_FLOATN z3 = __clc_mad(z, z2, a6);
      __CLC_FLOATN z4 = __clc_mad(z, z3, a4);
      __CLC_FLOATN z5 = __clc_mad(z, z4, a2);
      p1 = __clc_mad(z, z5, a0);

      __CLC_FLOATN z2b = __clc_mad(z, a11, a9);
      __CLC_FLOATN z3b = __clc_mad(z, z2b, a7);
      __CLC_FLOATN z4b = __clc_mad(z, z3b, a5);
      __CLC_FLOATN z5b = __clc_mad(z, z4b, a3);
      p2 = z * __clc_mad(z, z5b, a1);

      p = __clc_mad(y, p1, p2);
      ret += __clc_mad(y, -0.5f, p);
      break;
    }
    case 1: {
      z = y * y;
      w = z * y;

      __CLC_FLOATN w2 = __clc_mad(w, t12, t9);
      __CLC_FLOATN w3 = __clc_mad(w, w2, t6);
      __CLC_FLOATN w4 = __clc_mad(w, w3, t3);
      p1 = __clc_mad(w, w4, t0);

      __CLC_FLOATN w2b = __clc_mad(w, t13, t10);
      __CLC_FLOATN w3b = __clc_mad(w, w2b, t7);
      __CLC_FLOATN w4b = __clc_mad(w, w3b, t4);
      p2 = __clc_mad(w, w4b, t1);

      __CLC_FLOATN w2c = __clc_mad(w, t14, t11);
      __CLC_FLOATN w3c = __clc_mad(w, w2c, t8);
      __CLC_FLOATN w4c = __clc_mad(w, w3c, t5);
      p3 = __clc_mad(w, w4c, t2);

      __CLC_FLOATN negPart = -__clc_mad(w, -__clc_mad(y, p3, p2), tt);
      p = __clc_mad(z, p1, negPart);

      ret += tf + p;
      break;
    }
    case 2: {
      __CLC_FLOATN y2 = __clc_mad(y, u5, u4);
      __CLC_FLOATN y3 = __clc_mad(y, y2, u3);
      __CLC_FLOATN y4 = __clc_mad(y, y3, u2);
      __CLC_FLOATN y5 = __clc_mad(y, y4, u1);
      p1 = y * __clc_mad(y, y5, u0);

      __CLC_FLOATN y2b = __clc_mad(y, v5, v4);
      __CLC_FLOATN y3b = __clc_mad(y, y2b, v3);
      __CLC_FLOATN y4b = __clc_mad(y, y3b, v2);
      __CLC_FLOATN y5b = __clc_mad(y, y4b, v1);
      p2 = __clc_mad(y, y5b, 1.0f);

      ret += __clc_mad(y, -0.5f, __clc_div_fast(p1, p2));
      break;
    }
    }
  } else if (ax < 8.0f) { // 2 < |x| < 8
    __CLC_INTN i = __CLC_CONVERT_INTN(ax);
    __CLC_FLOATN y = ax - __CLC_CONVERT_FLOATN(i);

    __CLC_FLOATN p1 = __clc_mad(y, s6, s5);
    __CLC_FLOATN p2 = __clc_mad(y, p1, s4);
    __CLC_FLOATN p3 = __clc_mad(y, p2, s3);
    __CLC_FLOATN p4 = __clc_mad(y, p3, s2);
    __CLC_FLOATN p5 = __clc_mad(y, p4, s1);
    __CLC_FLOATN p = y * __clc_mad(y, p5, s0);

    __CLC_FLOATN q1 = __clc_mad(y, r6, r5);
    __CLC_FLOATN q2 = __clc_mad(y, q1, r4);
    __CLC_FLOATN q3 = __clc_mad(y, q2, r3);
    __CLC_FLOATN q4 = __clc_mad(y, q3, r2);
    __CLC_FLOATN q5 = __clc_mad(y, q4, r1);
    __CLC_FLOATN q = __clc_mad(y, q5, 1.0f);

    ret = __clc_mad(y, 0.5f, __clc_div_fast(p, q));

    __CLC_FLOATN y2 = y + 2.0f;
    __CLC_FLOATN y3 = y + 3.0f;
    __CLC_FLOATN y4 = y + 4.0f;
    __CLC_FLOATN y5 = y + 5.0f;
    __CLC_FLOATN y6 = y + 6.0f;

    __CLC_FLOATN z = 1.0f;
    z *= i > 2 ? y2 : 1.0f;
    z *= i > 3 ? y3 : 1.0f;
    z *= i > 4 ? y4 : 1.0f;
    z *= i > 5 ? y5 : 1.0f;
    z *= i > 6 ? y6 : 1.0f;

    ret += __clc_log(z);
  } else if (ax < 0x1.0p+58f) { // 8 <= |x| < 2^58
    __CLC_FLOATN z = __clc_recip_fast(ax);
    __CLC_FLOATN y = z * z;

    __CLC_FLOATN t1 = __clc_mad(y, w6, w5);
    __CLC_FLOATN t2 = __clc_mad(y, t1, w4);
    __CLC_FLOATN t3 = __clc_mad(y, t2, w3);
    __CLC_FLOATN t4 = __clc_mad(y, t3, w2);
    __CLC_FLOATN t5 = __clc_mad(y, t4, w1);
    __CLC_FLOATN w = __clc_mad(z, t5, w0);

    ret = __clc_mad(ax - 0.5f, __clc_log(ax) - 1.0f, w);
  } else {
    // 2^58 <= |x| <= Inf
    ret = __clc_mad(ax, __clc_log(ax), -ax);
  }

  __CLC_INTN s = 0;
  if (x >= 0.0f) {
    ret = ((x == 1.0f) | (x == 2.0f)) ? 0.0f : ret;
    s = x == 0.0f ? 0 : 1;
  } else if (ax < 0x1.0p+23f) { // x > -0x1.0p+23
    if (ax > 0x1.0p-21f) {
      __CLC_FLOATN t = __clc_sinpi(x);
      __CLC_FLOATN negadj = __clc_log(pi / __clc_fabs(t * x));
      ret = negadj - ret;

      bool z = __clc_trunc(x) == x;
      ret = z ? __CLC_GENTYPE_INF : ret;
      s = t < 0.0f ? -1 : 1;
      s = z ? 0 : s;
    } else {
      s = -1;
    }
  }

  ret = ((ax != 0.0f) && !__clc_isinf(ax) && ((x >= 0.0f) || (ax < 0x1.0p+23f)))
            ? ret
            : __CLC_GENTYPE_INF;

  ret = __clc_isnan(x) ? x : ret;

  __CLC_LGAMMA_R_RET_GENTYPE result;
  result.result = ret;
  result.sign = s;

  return result;
}

#elif __CLC_FPSIZE == 64

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_LGAMMA_R_RET_GENTYPE
__clc_lgamma_r_stret(__CLC_DOUBLEN x) {
  const __CLC_DOUBLEN pi = 3.14159265358979311600e+00;
  const __CLC_DOUBLEN a0 = 7.72156649015328655494e-02;
  const __CLC_DOUBLEN a1 = 3.22467033424113591611e-01;
  const __CLC_DOUBLEN a2 = 6.73523010531292681824e-02;
  const __CLC_DOUBLEN a3 = 2.05808084325167332806e-02;
  const __CLC_DOUBLEN a4 = 7.38555086081402883957e-03;
  const __CLC_DOUBLEN a5 = 2.89051383673415629091e-03;
  const __CLC_DOUBLEN a6 = 1.19270763183362067845e-03;
  const __CLC_DOUBLEN a7 = 5.10069792153511336608e-04;
  const __CLC_DOUBLEN a8 = 2.20862790713908385557e-04;
  const __CLC_DOUBLEN a9 = 1.08011567247583939954e-04;
  const __CLC_DOUBLEN a10 = 2.52144565451257326939e-05;
  const __CLC_DOUBLEN a11 = 4.48640949618915160150e-05;
  const __CLC_DOUBLEN tc = 1.46163214496836224576e+00;
  const __CLC_DOUBLEN tf = -1.21486290535849611461e-01;
  const __CLC_DOUBLEN tt = -3.63867699703950536541e-18;
  const __CLC_DOUBLEN t0 = 4.83836122723810047042e-01;
  const __CLC_DOUBLEN t1 = -1.47587722994593911752e-01;
  const __CLC_DOUBLEN t2 = 6.46249402391333854778e-02;
  const __CLC_DOUBLEN t3 = -3.27885410759859649565e-02;
  const __CLC_DOUBLEN t4 = 1.79706750811820387126e-02;
  const __CLC_DOUBLEN t5 = -1.03142241298341437450e-02;
  const __CLC_DOUBLEN t6 = 6.10053870246291332635e-03;
  const __CLC_DOUBLEN t7 = -3.68452016781138256760e-03;
  const __CLC_DOUBLEN t8 = 2.25964780900612472250e-03;
  const __CLC_DOUBLEN t9 = -1.40346469989232843813e-03;
  const __CLC_DOUBLEN t10 = 8.81081882437654011382e-04;
  const __CLC_DOUBLEN t11 = -5.38595305356740546715e-04;
  const __CLC_DOUBLEN t12 = 3.15632070903625950361e-04;
  const __CLC_DOUBLEN t13 = -3.12754168375120860518e-04;
  const __CLC_DOUBLEN t14 = 3.35529192635519073543e-04;
  const __CLC_DOUBLEN u0 = -7.72156649015328655494e-02;
  const __CLC_DOUBLEN u1 = 6.32827064025093366517e-01;
  const __CLC_DOUBLEN u2 = 1.45492250137234768737e+00;
  const __CLC_DOUBLEN u3 = 9.77717527963372745603e-01;
  const __CLC_DOUBLEN u4 = 2.28963728064692451092e-01;
  const __CLC_DOUBLEN u5 = 1.33810918536787660377e-02;
  const __CLC_DOUBLEN v1 = 2.45597793713041134822e+00;
  const __CLC_DOUBLEN v2 = 2.12848976379893395361e+00;
  const __CLC_DOUBLEN v3 = 7.69285150456672783825e-01;
  const __CLC_DOUBLEN v4 = 1.04222645593369134254e-01;
  const __CLC_DOUBLEN v5 = 3.21709242282423911810e-03;
  const __CLC_DOUBLEN s0 = -7.72156649015328655494e-02;
  const __CLC_DOUBLEN s1 = 2.14982415960608852501e-01;
  const __CLC_DOUBLEN s2 = 3.25778796408930981787e-01;
  const __CLC_DOUBLEN s3 = 1.46350472652464452805e-01;
  const __CLC_DOUBLEN s4 = 2.66422703033638609560e-02;
  const __CLC_DOUBLEN s5 = 1.84028451407337715652e-03;
  const __CLC_DOUBLEN s6 = 3.19475326584100867617e-05;
  const __CLC_DOUBLEN r1 = 1.39200533467621045958e+00;
  const __CLC_DOUBLEN r2 = 7.21935547567138069525e-01;
  const __CLC_DOUBLEN r3 = 1.71933865632803078993e-01;
  const __CLC_DOUBLEN r4 = 1.86459191715652901344e-02;
  const __CLC_DOUBLEN r5 = 7.77942496381893596434e-04;
  const __CLC_DOUBLEN r6 = 7.32668430744625636189e-06;
  const __CLC_DOUBLEN w0 = 4.18938533204672725052e-01;
  const __CLC_DOUBLEN w1 = 8.33333333333329678849e-02;
  const __CLC_DOUBLEN w2 = -2.77777777728775536470e-03;
  const __CLC_DOUBLEN w3 = 7.93650558643019558500e-04;
  const __CLC_DOUBLEN w4 = -5.95187557450339963135e-04;
  const __CLC_DOUBLEN w5 = 8.36339918996282139126e-04;
  const __CLC_DOUBLEN w6 = -1.63092934096575273989e-03;
  const __CLC_DOUBLEN z1 = -0x1.2788cfc6fb619p-1;
  const __CLC_DOUBLEN z2 = 0x1.a51a6625307d3p-1;
  const __CLC_DOUBLEN z3 = -0x1.9a4d55beab2d7p-2;
  const __CLC_DOUBLEN z4 = 0x1.151322ac7d848p-2;
  const __CLC_DOUBLEN z5 = -0x1.a8b9c17aa6149p-3;

  __CLC_DOUBLEN ax = __clc_fabs(x);
  __CLC_DOUBLEN ret;

  if (ax < 0x1.0p-8) {
    __CLC_DOUBLEN t1 = __clc_mad(ax, z5, z4);
    __CLC_DOUBLEN t2 = __clc_mad(ax, t1, z3);
    __CLC_DOUBLEN t3 = __clc_mad(ax, t2, z2);
    __CLC_DOUBLEN t4 = __clc_mad(ax, t3, z1);
    ret = __clc_mad(ax, t4, -__clc_log(ax));
  } else if (ax < 2.0) {
    __CLC_INTN i;
    bool c;
    __CLC_DOUBLEN y, t;
    if (ax <= 0x1.cccccp-1) { // |x| < 0.9 : lgamma(x) = lgamma(x+1)-log(x)
      ret = -__clc_log(ax);

      y = 1.0 - ax;
      i = 0;

      c = ax < 0x1.76944p-1; // x < 0.7316
      t = ax - (tc - 1.0);
      y = c ? t : y;
      i = c ? 1 : i;

      c = ax < 0x1.da661p-3; // x < .2316
      y = c ? ax : y;
      i = c ? 2 : i;
    } else {
      ret = 0.0;

      y = 2.0 - ax;
      i = 0;

      c = ax < 0x1.bb4c3p+0; // x < 1.7316
      t = ax - tc;
      y = c ? t : y;
      i = c ? 1 : i;

      c = ax < 0x1.3b4c4p+0; // x < 1.2316
      t = ax - 1.0;
      y = c ? t : y;
      i = c ? 2 : i;
    }

    __CLC_DOUBLEN w, z, p, p1, p2, p3;
    switch (i) {
    case 0: {
      z = y * y;

      __CLC_DOUBLEN z2 = __clc_mad(z, a10, a8);
      __CLC_DOUBLEN z3 = __clc_mad(z, z2, a6);
      __CLC_DOUBLEN z4 = __clc_mad(z, z3, a4);
      __CLC_DOUBLEN z5 = __clc_mad(z, z4, a2);
      p1 = __clc_mad(z, z5, a0);

      __CLC_DOUBLEN z2b = __clc_mad(z, a11, a9);
      __CLC_DOUBLEN z3b = __clc_mad(z, z2b, a7);
      __CLC_DOUBLEN z4b = __clc_mad(z, z3b, a5);
      __CLC_DOUBLEN z5b = __clc_mad(z, z4b, a3);
      p2 = z * __clc_mad(z, z5b, a1);

      p = __clc_mad(y, p1, p2);
      ret += __clc_mad(y, -0.5, p);
      break;
    }
    case 1: {
      z = y * y;
      w = z * y;

      __CLC_DOUBLEN w2 = __clc_mad(w, t12, t9);
      __CLC_DOUBLEN w3 = __clc_mad(w, w2, t6);
      __CLC_DOUBLEN w4 = __clc_mad(w, w3, t3);
      p1 = __clc_mad(w, w4, t0);

      __CLC_DOUBLEN w2b = __clc_mad(w, t13, t10);
      __CLC_DOUBLEN w3b = __clc_mad(w, w2b, t7);
      __CLC_DOUBLEN w4b = __clc_mad(w, w3b, t4);
      p2 = __clc_mad(w, w4b, t1);

      __CLC_DOUBLEN w2c = __clc_mad(w, t14, t11);
      __CLC_DOUBLEN w3c = __clc_mad(w, w2c, t8);
      __CLC_DOUBLEN w4c = __clc_mad(w, w3c, t5);
      p3 = __clc_mad(w, w4c, t2);

      __CLC_DOUBLEN inner = __clc_mad(y, p3, p2);
      __CLC_DOUBLEN negPart = -__clc_mad(w, -inner, tt);
      p = __clc_mad(z, p1, negPart);

      ret += tf + p;
      break;
    }
    case 2: {
      __CLC_DOUBLEN y2 = __clc_mad(y, u5, u4);
      __CLC_DOUBLEN y3 = __clc_mad(y, y2, u3);
      __CLC_DOUBLEN y4 = __clc_mad(y, y3, u2);
      __CLC_DOUBLEN y5 = __clc_mad(y, y4, u1);
      p1 = y * __clc_mad(y, y5, u0);

      __CLC_DOUBLEN y2b = __clc_mad(y, v5, v4);
      __CLC_DOUBLEN y3b = __clc_mad(y, y2b, v3);
      __CLC_DOUBLEN y4b = __clc_mad(y, y3b, v2);
      __CLC_DOUBLEN y5b = __clc_mad(y, y4b, v1);
      p2 = __clc_mad(y, y5b, 1.0);

      ret += __clc_mad(y, -0.5, p1 / p2);
      break;
    }
    }
  } else if (ax < 8.0) { // 2 < ax < 8
    __CLC_INTN i = __CLC_CONVERT_INTN(ax);
    __CLC_DOUBLEN y = ax - __CLC_CONVERT_DOUBLEN(i);

    __CLC_DOUBLEN p1 = __clc_mad(y, s6, s5);
    __CLC_DOUBLEN p2 = __clc_mad(y, p1, s4);
    __CLC_DOUBLEN p3 = __clc_mad(y, p2, s3);
    __CLC_DOUBLEN p4 = __clc_mad(y, p3, s2);
    __CLC_DOUBLEN p5 = __clc_mad(y, p4, s1);
    __CLC_DOUBLEN p = y * __clc_mad(y, p5, s0);

    __CLC_DOUBLEN q1 = __clc_mad(y, r6, r5);
    __CLC_DOUBLEN q2 = __clc_mad(y, q1, r4);
    __CLC_DOUBLEN q3 = __clc_mad(y, q2, r3);
    __CLC_DOUBLEN q4 = __clc_mad(y, q3, r2);
    __CLC_DOUBLEN q5 = __clc_mad(y, q4, r1);
    __CLC_DOUBLEN q = __clc_mad(y, q5, 1.0);

    ret = __clc_mad(y, 0.5, p / q);

    __CLC_DOUBLEN y2 = y + 2.0;
    __CLC_DOUBLEN y3 = y + 3.0;
    __CLC_DOUBLEN y4 = y + 4.0;
    __CLC_DOUBLEN y5 = y + 5.0;
    __CLC_DOUBLEN y6 = y + 6.0;

    __CLC_DOUBLEN z = 1.0;
    z *= i > 2 ? y2 : 1.0;
    z *= i > 3 ? y3 : 1.0;
    z *= i > 4 ? y4 : 1.0;
    z *= i > 5 ? y5 : 1.0;
    z *= i > 6 ? y6 : 1.0;

    ret += __clc_log(z);
  } else if (ax < 0x1p+58) { // 8 <= ax < 2^58
    __CLC_DOUBLEN z = 1.0 / ax;
    __CLC_DOUBLEN y = z * z;

    // Nested multiply-add expansions
    __CLC_DOUBLEN t1 = __clc_mad(y, w6, w5);
    __CLC_DOUBLEN t2 = __clc_mad(y, t1, w4);
    __CLC_DOUBLEN t3 = __clc_mad(y, t2, w3);
    __CLC_DOUBLEN t4 = __clc_mad(y, t3, w2);
    __CLC_DOUBLEN t5 = __clc_mad(y, t4, w1);
    __CLC_DOUBLEN w = __clc_mad(z, t5, w0);

    ret = __clc_mad(ax - 0.5, __clc_log(ax) - 1.0, w);
  } else { // 2^58 <= ax <= Inf
    ret = __clc_mad(ax, __clc_log(ax), -ax);
  }

  __CLC_INTN s = 0;
  if (x >= 0.0) {
    ret = (x == 1.0 | x == 2.0) ? 0.0 : ret;
    s = x == 0.0 ? 0 : 1;
  } else if (ax < 0x1p+52) { // x > -0x1.0p+52
    if (ax > 0x1.0p-50) {    // x < -0x1.0p-50
      __CLC_DOUBLEN t = __clc_sinpi(x);
      __CLC_DOUBLEN negadj = __clc_log(pi / __clc_fabs(t * x));
      ret = negadj - ret;

      bool z = __clc_trunc(x) == x;
      ret = z ? __CLC_GENTYPE_INF : ret;
      s = t < 0.0 ? -1 : 1;
      s = z ? 0 : s;
    } else {
      s = -1;
    }
  }

  // Handle negative integer, Inf, NaN
  ret = (ax == 0.0 || ax == INFINITY) || (x < 0.0 & ax >= 0x1p+52)
            ? __CLC_GENTYPE_INF
            : ret;
  ret = __clc_isnan(x) ? x : ret;

  __CLC_LGAMMA_R_RET_GENTYPE result;
  result.result = ret;
  result.sign = s;
  return result;
}

#elif __CLC_FPSIZE == 16

_CLC_DEF _CLC_OVERLOAD _CLC_CONST __CLC_LGAMMA_R_RET_GENTYPE
__clc_lgamma_r_stret(__CLC_HALFN x) {
  __CLC_FLOATN x_promoted = __CLC_CONVERT_FLOATN(x);
  __CLC_XCONCAT(__clc_lgamma_r_ret_, __CLC_FLOATN)
  promoted_result = __clc_lgamma_r_stret(x_promoted);

  __CLC_LGAMMA_R_RET_GENTYPE result = {
      __CLC_CONVERT_HALFN(promoted_result.result), promoted_result.sign};
  return result;
}

#endif

#endif // __CLC_SCALAR
