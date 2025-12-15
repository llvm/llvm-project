//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <clc/clc_convert.h>
#include <clc/float/definitions.h>
#include <clc/internal/clc.h>
#include <clc/math/clc_fabs.h>
#include <clc/math/clc_fma.h>
#include <clc/math/clc_log.h>
#include <clc/math/clc_mad.h>
#include <clc/math/clc_sinpi.h>
#include <clc/math/math.h>

// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

#define pi_f 3.1415927410e+00f /* 0x40490fdb */

#define a0_f 7.7215664089e-02f  /* 0x3d9e233f */
#define a1_f 3.2246702909e-01f  /* 0x3ea51a66 */
#define a2_f 6.7352302372e-02f  /* 0x3d89f001 */
#define a3_f 2.0580807701e-02f  /* 0x3ca89915 */
#define a4_f 7.3855509982e-03f  /* 0x3bf2027e */
#define a5_f 2.8905137442e-03f  /* 0x3b3d6ec6 */
#define a6_f 1.1927076848e-03f  /* 0x3a9c54a1 */
#define a7_f 5.1006977446e-04f  /* 0x3a05b634 */
#define a8_f 2.2086278477e-04f  /* 0x39679767 */
#define a9_f 1.0801156895e-04f  /* 0x38e28445 */
#define a10_f 2.5214456400e-05f /* 0x37d383a2 */
#define a11_f 4.4864096708e-05f /* 0x383c2c75 */

#define tc_f 1.4616321325e+00f /* 0x3fbb16c3 */

#define tf_f -1.2148628384e-01f /* 0xbdf8cdcd */
/* tt -(tail of tf) */
#define tt_f 6.6971006518e-09f /* 0x31e61c52 */

#define t0_f 4.8383611441e-01f   /* 0x3ef7b95e */
#define t1_f -1.4758771658e-01f  /* 0xbe17213c */
#define t2_f 6.4624942839e-02f   /* 0x3d845a15 */
#define t3_f -3.2788541168e-02f  /* 0xbd064d47 */
#define t4_f 1.7970675603e-02f   /* 0x3c93373d */
#define t5_f -1.0314224288e-02f  /* 0xbc28fcfe */
#define t6_f 6.1005386524e-03f   /* 0x3bc7e707 */
#define t7_f -3.6845202558e-03f  /* 0xbb7177fe */
#define t8_f 2.2596477065e-03f   /* 0x3b141699 */
#define t9_f -1.4034647029e-03f  /* 0xbab7f476 */
#define t10_f 8.8108185446e-04f  /* 0x3a66f867 */
#define t11_f -5.3859531181e-04f /* 0xba0d3085 */
#define t12_f 3.1563205994e-04f  /* 0x39a57b6b */
#define t13_f -3.1275415677e-04f /* 0xb9a3f927 */
#define t14_f 3.3552918467e-04f  /* 0x39afe9f7 */

#define u0_f -7.7215664089e-02f /* 0xbd9e233f */
#define u1_f 6.3282704353e-01f  /* 0x3f2200f4 */
#define u2_f 1.4549225569e+00f  /* 0x3fba3ae7 */
#define u3_f 9.7771751881e-01f  /* 0x3f7a4bb2 */
#define u4_f 2.2896373272e-01f  /* 0x3e6a7578 */
#define u5_f 1.3381091878e-02f  /* 0x3c5b3c5e */

#define v1_f 2.4559779167e+00f /* 0x401d2ebe */
#define v2_f 2.1284897327e+00f /* 0x4008392d */
#define v3_f 7.6928514242e-01f /* 0x3f44efdf */
#define v4_f 1.0422264785e-01f /* 0x3dd572af */
#define v5_f 3.2170924824e-03f /* 0x3b52d5db */

#define s0_f -7.7215664089e-02f /* 0xbd9e233f */
#define s1_f 2.1498242021e-01f  /* 0x3e5c245a */
#define s2_f 3.2577878237e-01f  /* 0x3ea6cc7a */
#define s3_f 1.4635047317e-01f  /* 0x3e15dce6 */
#define s4_f 2.6642270386e-02f  /* 0x3cda40e4 */
#define s5_f 1.8402845599e-03f  /* 0x3af135b4 */
#define s6_f 3.1947532989e-05f  /* 0x3805ff67 */

#define r1_f 1.3920053244e+00f /* 0x3fb22d3b */
#define r2_f 7.2193557024e-01f /* 0x3f38d0c5 */
#define r3_f 1.7193385959e-01f /* 0x3e300f6e */
#define r4_f 1.8645919859e-02f /* 0x3c98bf54 */
#define r5_f 7.7794247773e-04f /* 0x3a4beed6 */
#define r6_f 7.3266842264e-06f /* 0x36f5d7bd */

#define w0_f 4.1893854737e-01f  /* 0x3ed67f1d */
#define w1_f 8.3333335817e-02f  /* 0x3daaaaab */
#define w2_f -2.7777778450e-03f /* 0xbb360b61 */
#define w3_f 7.9365057172e-04f  /* 0x3a500cfd */
#define w4_f -5.9518753551e-04f /* 0xba1c065c */
#define w5_f 8.3633989561e-04f  /* 0x3a5b3dd2 */
#define w6_f -1.6309292987e-03f /* 0xbad5c4e8 */

_CLC_OVERLOAD _CLC_DEF float __clc_lgamma_r(float x, private int *signp) {
  int hx = __clc_as_int(x);
  float absx = __clc_fabs(x);
  int ix = __clc_as_int(absx);

  if (ix >= 0x7f800000) {
    *signp = 1;
    return x;
  }

  if (absx < 0x1.0p-70f) {
    *signp = hx < 0 ? -1 : 1;
    return -__clc_log(absx);
  }

  float r;

  if (absx == 1.0f | absx == 2.0f)
    r = 0.0f;

  else if (absx < 2.0f) {
    float y = 2.0f - absx;
    int i = 0;

    int c = absx < 0x1.bb4c30p+0f;
    float yt = absx - tc_f;
    y = c ? yt : y;
    i = c ? 1 : i;

    c = absx < 0x1.3b4c40p+0f;
    yt = absx - 1.0f;
    y = c ? yt : y;
    i = c ? 2 : i;

    r = -__clc_log(absx);
    yt = 1.0f - absx;
    c = absx <= 0x1.ccccccp-1f;
    r = c ? r : 0.0f;
    y = c ? yt : y;
    i = c ? 0 : i;

    c = absx < 0x1.769440p-1f;
    yt = absx - (tc_f - 1.0f);
    y = c ? yt : y;
    i = c ? 1 : i;

    c = absx < 0x1.da6610p-3f;
    y = c ? absx : y;
    i = c ? 2 : i;

    float z, w, p1, p2, p3, p;
    switch (i) {
    case 0:
      z = y * y;
      p1 = __clc_mad(
          z,
          __clc_mad(
              z,
              __clc_mad(z, __clc_mad(z, __clc_mad(z, a10_f, a8_f), a6_f), a4_f),
              a2_f),
          a0_f);
      p2 = z *
           __clc_mad(
               z,
               __clc_mad(
                   z,
                   __clc_mad(z, __clc_mad(z, __clc_mad(z, a11_f, a9_f), a7_f),
                             a5_f),
                   a3_f),
               a1_f);
      p = __clc_mad(y, p1, p2);
      r += __clc_mad(y, -0.5f, p);
      break;
    case 1:
      z = y * y;
      w = z * y;
      p1 = __clc_mad(
          w, __clc_mad(w, __clc_mad(w, __clc_mad(w, t12_f, t9_f), t6_f), t3_f),
          t0_f);
      p2 = __clc_mad(
          w, __clc_mad(w, __clc_mad(w, __clc_mad(w, t13_f, t10_f), t7_f), t4_f),
          t1_f);
      p3 = __clc_mad(
          w, __clc_mad(w, __clc_mad(w, __clc_mad(w, t14_f, t11_f), t8_f), t5_f),
          t2_f);
      p = __clc_mad(z, p1, -__clc_mad(w, -__clc_mad(y, p3, p2), tt_f));
      r += tf_f + p;
      break;
    case 2:
      p1 = y *
           __clc_mad(
               y,
               __clc_mad(y,
                         __clc_mad(y,
                                   __clc_mad(y, __clc_mad(y, u5_f, u4_f), u3_f),
                                   u2_f),
                         u1_f),
               u0_f);
      p2 = __clc_mad(
          y,
          __clc_mad(
              y,
              __clc_mad(y, __clc_mad(y, __clc_mad(y, v5_f, v4_f), v3_f), v2_f),
              v1_f),
          1.0f);
      r += __clc_mad(y, -0.5f, MATH_DIVIDE(p1, p2));
      break;
    }
  } else if (absx < 8.0f) {
    int i = (int)absx;
    float y = absx - (float)i;
    float p =
        y *
        __clc_mad(
            y,
            __clc_mad(
                y,
                __clc_mad(
                    y,
                    __clc_mad(y, __clc_mad(y, __clc_mad(y, s6_f, s5_f), s4_f),
                              s3_f),
                    s2_f),
                s1_f),
            s0_f);
    float q = __clc_mad(
        y,
        __clc_mad(
            y,
            __clc_mad(y,
                      __clc_mad(y, __clc_mad(y, __clc_mad(y, r6_f, r5_f), r4_f),
                                r3_f),
                      r2_f),
            r1_f),
        1.0f);
    r = __clc_mad(y, 0.5f, MATH_DIVIDE(p, q));

    float y6 = y + 6.0f;
    float y5 = y + 5.0f;
    float y4 = y + 4.0f;
    float y3 = y + 3.0f;
    float y2 = y + 2.0f;

    float z = 1.0f;
    z *= i > 6 ? y6 : 1.0f;
    z *= i > 5 ? y5 : 1.0f;
    z *= i > 4 ? y4 : 1.0f;
    z *= i > 3 ? y3 : 1.0f;
    z *= i > 2 ? y2 : 1.0f;

    r += __clc_log(z);
  } else if (absx < 0x1.0p+58f) {
    float z = 1.0f / absx;
    float y = z * z;
    float w = __clc_mad(
        z,
        __clc_mad(
            y,
            __clc_mad(y,
                      __clc_mad(y, __clc_mad(y, __clc_mad(y, w6_f, w5_f), w4_f),
                                w3_f),
                      w2_f),
            w1_f),
        w0_f);
    r = __clc_mad(absx - 0.5f, __clc_log(absx) - 1.0f, w);
  } else
    // 2**58 <= x <= Inf
    r = absx * (__clc_log(absx) - 1.0f);

  int s = 1;

  if (x < 0.0f) {
    float t = __clc_sinpi(x);
    r = __clc_log(pi_f / __clc_fabs(t * x)) - r;
    r = t == 0.0f ? INFINITY : r;
    s = t < 0.0f ? -1 : s;
  }

  *signp = s;
  return r;
}

#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
// ====================================================
// Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
// Developed at SunPro, a Sun Microsystems, Inc. business.
// Permission to use, copy, modify, and distribute this
// software is freely granted, provided that this notice
// is preserved.
// ====================================================

// lgamma_r(x, i)
// Reentrant version of the logarithm of the Gamma function
// with user provide pointer for the sign of Gamma(x).
//
// Method:
//   1. Argument Reduction for 0 < x <= 8
//      Since gamma(1+s)=s*gamma(s), for x in [0,8], we may
//      reduce x to a number in [1.5,2.5] by
//              lgamma(1+s) = log(s) + lgamma(s)
//      for example,
//              lgamma(7.3) = log(6.3) + lgamma(6.3)
//                          = log(6.3*5.3) + lgamma(5.3)
//                          = log(6.3*5.3*4.3*3.3*2.3) + lgamma(2.3)
//   2. Polynomial approximation of lgamma around its
//      minimun ymin=1.461632144968362245 to maintain monotonicity.
//      On [ymin-0.23, ymin+0.27] (i.e., [1.23164,1.73163]), use
//              Let z = x-ymin;
//              lgamma(x) = -1.214862905358496078218 + z^2*poly(z)
//      where
//              poly(z) is a 14 degree polynomial.
//   2. Rational approximation in the primary interval [2,3]
//      We use the following approximation:
//              s = x-2.0;
//              lgamma(x) = 0.5*s + s*P(s)/Q(s)
//      with accuracy
//              |P/Q - (lgamma(x)-0.5s)| < 2**-61.71
//      Our algorithms are based on the following observation
//
//                             zeta(2)-1    2    zeta(3)-1    3
// lgamma(2+s) = s*(1-Euler) + --------- * s  -  --------- * s  + ...
//                                 2                 3
//
//      where Euler = 0.5771... is the Euler constant, which is very
//      close to 0.5.
//
//   3. For x>=8, we have
//      lgamma(x)~(x-0.5)log(x)-x+0.5*log(2pi)+1/(12x)-1/(360x**3)+....
//      (better formula:
//         lgamma(x)~(x-0.5)*(log(x)-1)-.5*(log(2pi)-1) + ...)
//      Let z = 1/x, then we approximation
//              f(z) = lgamma(x) - (x-0.5)(log(x)-1)
//      by
//                                  3       5             11
//              w = w0 + w1*z + w2*z  + w3*z  + ... + w6*z
//      where
//              |w - f(z)| < 2**-58.74
//
//   4. For negative x, since (G is gamma function)
//              -x*G(-x)*G(x) = pi/sin(pi*x),
//      we have
//              G(x) = pi/(sin(pi*x)*(-x)*G(-x))
//      since G(-x) is positive, sign(G(x)) = sign(sin(pi*x)) for x<0
//      Hence, for x<0, signgam = sign(sin(pi*x)) and
//              lgamma(x) = log(|Gamma(x)|)
//                        = log(pi/(|x*sin(pi*x)|)) - lgamma(-x);
//      Note: one should avoid compute pi*(-x) directly in the
//            computation of sin(pi*(-x)).
//
//   5. Special Cases
//              lgamma(2+s) ~ s*(1-Euler) for tiny s
//              lgamma(1)=lgamma(2)=0
//              lgamma(x) ~ -log(x) for tiny x
//              lgamma(0) = lgamma(inf) = inf
//              lgamma(-integer) = +-inf
//
#define pi 3.14159265358979311600e+00 /* 0x400921FB, 0x54442D18 */

#define a0 7.72156649015328655494e-02  /* 0x3FB3C467, 0xE37DB0C8 */
#define a1 3.22467033424113591611e-01  /* 0x3FD4A34C, 0xC4A60FAD */
#define a2 6.73523010531292681824e-02  /* 0x3FB13E00, 0x1A5562A7 */
#define a3 2.05808084325167332806e-02  /* 0x3F951322, 0xAC92547B */
#define a4 7.38555086081402883957e-03  /* 0x3F7E404F, 0xB68FEFE8 */
#define a5 2.89051383673415629091e-03  /* 0x3F67ADD8, 0xCCB7926B */
#define a6 1.19270763183362067845e-03  /* 0x3F538A94, 0x116F3F5D */
#define a7 5.10069792153511336608e-04  /* 0x3F40B6C6, 0x89B99C00 */
#define a8 2.20862790713908385557e-04  /* 0x3F2CF2EC, 0xED10E54D */
#define a9 1.08011567247583939954e-04  /* 0x3F1C5088, 0x987DFB07 */
#define a10 2.52144565451257326939e-05 /* 0x3EFA7074, 0x428CFA52 */
#define a11 4.48640949618915160150e-05 /* 0x3F07858E, 0x90A45837 */

#define tc 1.46163214496836224576e+00  /* 0x3FF762D8, 0x6356BE3F */
#define tf -1.21486290535849611461e-01 /* 0xBFBF19B9, 0xBCC38A42 */
#define tt -3.63867699703950536541e-18 /* 0xBC50C7CA, 0xA48A971F */

#define t0 4.83836122723810047042e-01   /* 0x3FDEF72B, 0xC8EE38A2 */
#define t1 -1.47587722994593911752e-01  /* 0xBFC2E427, 0x8DC6C509 */
#define t2 6.46249402391333854778e-02   /* 0x3FB08B42, 0x94D5419B */
#define t3 -3.27885410759859649565e-02  /* 0xBFA0C9A8, 0xDF35B713 */
#define t4 1.79706750811820387126e-02   /* 0x3F9266E7, 0x970AF9EC */
#define t5 -1.03142241298341437450e-02  /* 0xBF851F9F, 0xBA91EC6A */
#define t6 6.10053870246291332635e-03   /* 0x3F78FCE0, 0xE370E344 */
#define t7 -3.68452016781138256760e-03  /* 0xBF6E2EFF, 0xB3E914D7 */
#define t8 2.25964780900612472250e-03   /* 0x3F6282D3, 0x2E15C915 */
#define t9 -1.40346469989232843813e-03  /* 0xBF56FE8E, 0xBF2D1AF1 */
#define t10 8.81081882437654011382e-04  /* 0x3F4CDF0C, 0xEF61A8E9 */
#define t11 -5.38595305356740546715e-04 /* 0xBF41A610, 0x9C73E0EC */
#define t12 3.15632070903625950361e-04  /* 0x3F34AF6D, 0x6C0EBBF7 */
#define t13 -3.12754168375120860518e-04 /* 0xBF347F24, 0xECC38C38 */
#define t14 3.35529192635519073543e-04  /* 0x3F35FD3E, 0xE8C2D3F4 */

#define u0 -7.72156649015328655494e-02 /* 0xBFB3C467, 0xE37DB0C8 */
#define u1 6.32827064025093366517e-01  /* 0x3FE4401E, 0x8B005DFF */
#define u2 1.45492250137234768737e+00  /* 0x3FF7475C, 0xD119BD6F */
#define u3 9.77717527963372745603e-01  /* 0x3FEF4976, 0x44EA8450 */
#define u4 2.28963728064692451092e-01  /* 0x3FCD4EAE, 0xF6010924 */
#define u5 1.33810918536787660377e-02  /* 0x3F8B678B, 0xBF2BAB09 */

#define v1 2.45597793713041134822e+00 /* 0x4003A5D7, 0xC2BD619C */
#define v2 2.12848976379893395361e+00 /* 0x40010725, 0xA42B18F5 */
#define v3 7.69285150456672783825e-01 /* 0x3FE89DFB, 0xE45050AF */
#define v4 1.04222645593369134254e-01 /* 0x3FBAAE55, 0xD6537C88 */
#define v5 3.21709242282423911810e-03 /* 0x3F6A5ABB, 0x57D0CF61 */

#define s0_d -7.72156649015328655494e-02 /* 0xBFB3C467, 0xE37DB0C8 */
#define s1_d 2.14982415960608852501e-01  /* 0x3FCB848B, 0x36E20878 */
#define s2_d 3.25778796408930981787e-01  /* 0x3FD4D98F, 0x4F139F59 */
#define s3_d 1.46350472652464452805e-01  /* 0x3FC2BB9C, 0xBEE5F2F7 */
#define s4_d 2.66422703033638609560e-02  /* 0x3F9B481C, 0x7E939961 */
#define s5_d 1.84028451407337715652e-03  /* 0x3F5E26B6, 0x7368F239 */
#define s6_d 3.19475326584100867617e-05  /* 0x3F00BFEC, 0xDD17E945 */

#define r1 1.39200533467621045958e+00 /* 0x3FF645A7, 0x62C4AB74 */
#define r2 7.21935547567138069525e-01 /* 0x3FE71A18, 0x93D3DCDC */
#define r3 1.71933865632803078993e-01 /* 0x3FC601ED, 0xCCFBDF27 */
#define r4 1.86459191715652901344e-02 /* 0x3F9317EA, 0x742ED475 */
#define r5 7.77942496381893596434e-04 /* 0x3F497DDA, 0xCA41A95B */
#define r6 7.32668430744625636189e-06 /* 0x3EDEBAF7, 0xA5B38140 */

#define w0 4.18938533204672725052e-01  /* 0x3FDACFE3, 0x90C97D69 */
#define w1 8.33333333333329678849e-02  /* 0x3FB55555, 0x5555553B */
#define w2 -2.77777777728775536470e-03 /* 0xBF66C16C, 0x16B02E5C */
#define w3 7.93650558643019558500e-04  /* 0x3F4A019F, 0x98CF38B6 */
#define w4 -5.95187557450339963135e-04 /* 0xBF4380CB, 0x8C0FE741 */
#define w5 8.36339918996282139126e-04  /* 0x3F4B67BA, 0x4CDAD5D1 */
#define w6 -1.63092934096575273989e-03 /* 0xBF5AB89D, 0x0B9E43E4 */

_CLC_OVERLOAD _CLC_DEF double __clc_lgamma_r(double x, private int *ip) {
  ulong ux = __clc_as_ulong(x);
  double absx = __clc_fabs(x);
  ulong ax = __clc_as_ulong(absx);

  if (ax >= 0x7ff0000000000000UL) {
    // +-Inf, NaN
    *ip = 1;
    return absx;
  }

  if (absx < 0x1.0p-70) {
    *ip = ax == ux ? 1 : -1;
    return -__clc_log(absx);
  }

  // Handle rest of range
  double r;

  if (absx < 2.0) {
    int i = 0;
    double y = 2.0 - absx;

    int c = absx < 0x1.bb4c3p+0;
    double t = absx - tc;
    i = c ? 1 : i;
    y = c ? t : y;

    c = absx < 0x1.3b4c4p+0;
    t = absx - 1.0;
    i = c ? 2 : i;
    y = c ? t : y;

    c = absx <= 0x1.cccccp-1;
    t = -__clc_log(absx);
    r = c ? t : 0.0;
    t = 1.0 - absx;
    i = c ? 0 : i;
    y = c ? t : y;

    c = absx < 0x1.76944p-1;
    t = absx - (tc - 1.0);
    i = c ? 1 : i;
    y = c ? t : y;

    c = absx < 0x1.da661p-3;
    i = c ? 2 : i;
    y = c ? absx : y;

    double p, q;

    switch (i) {
    case 0:
      p = __clc_fma(
          y, __clc_fma(y, __clc_fma(y, __clc_fma(y, a11, a10), a9), a8), a7);
      p = __clc_fma(y, __clc_fma(y, __clc_fma(y, __clc_fma(y, p, a6), a5), a4),
                    a3);
      p = __clc_fma(y, __clc_fma(y, __clc_fma(y, p, a2), a1), a0);
      r = __clc_fma(y, p - 0.5, r);
      break;
    case 1:
      p = __clc_fma(
          y, __clc_fma(y, __clc_fma(y, __clc_fma(y, t14, t13), t12), t11), t10);
      p = __clc_fma(
          y,
          __clc_fma(y, __clc_fma(y, __clc_fma(y, __clc_fma(y, p, t9), t8), t7),
                    t6),
          t5);
      p = __clc_fma(
          y,
          __clc_fma(y, __clc_fma(y, __clc_fma(y, __clc_fma(y, p, t4), t3), t2),
                    t1),
          t0);
      p = __clc_fma(y * y, p, -tt);
      r += (tf + p);
      break;
    case 2:
      p = y *
          __clc_fma(
              y,
              __clc_fma(
                  y, __clc_fma(y, __clc_fma(y, __clc_fma(y, u5, u4), u3), u2),
                  u1),
              u0);
      q = __clc_fma(
          y,
          __clc_fma(y, __clc_fma(y, __clc_fma(y, __clc_fma(y, v5, v4), v3), v2),
                    v1),
          1.0);
      r += __clc_fma(-0.5, y, p / q);
    }
  } else if (absx < 8.0) {
    int i = absx;
    double y = absx - (double)i;
    double p =
        y *
        __clc_fma(
            y,
            __clc_fma(
                y,
                __clc_fma(
                    y,
                    __clc_fma(y, __clc_fma(y, __clc_fma(y, s6_d, s5_d), s4_d),
                              s3_d),
                    s2_d),
                s1_d),
            s0_d);
    double q = __clc_fma(
        y,
        __clc_fma(
            y,
            __clc_fma(y,
                      __clc_fma(y, __clc_fma(y, __clc_fma(y, r6, r5), r4), r3),
                      r2),
            r1),
        1.0);
    r = __clc_fma(0.5, y, p / q);
    double z = 1.0;
    // lgamma(1+s) = log(s) + lgamma(s)
    double y6 = y + 6.0;
    double y5 = y + 5.0;
    double y4 = y + 4.0;
    double y3 = y + 3.0;
    double y2 = y + 2.0;
    z *= i > 6 ? y6 : 1.0;
    z *= i > 5 ? y5 : 1.0;
    z *= i > 4 ? y4 : 1.0;
    z *= i > 3 ? y3 : 1.0;
    z *= i > 2 ? y2 : 1.0;
    r += __clc_log(z);
  } else {
    double z = 1.0 / absx;
    double z2 = z * z;
    double w = __clc_fma(
        z,
        __clc_fma(
            z2,
            __clc_fma(
                z2, __clc_fma(z2, __clc_fma(z2, __clc_fma(z2, w6, w5), w4), w3),
                w2),
            w1),
        w0);
    r = (absx - 0.5) * (__clc_log(absx) - 1.0) + w;
  }

  if (x < 0.0) {
    double t = __clc_sinpi(x);
    r = __clc_log(pi / __clc_fabs(t * x)) - r;
    r = t == 0.0 ? INFINITY : r;
    *ip = t < 0.0 ? -1 : 1;
  } else
    *ip = 1;

  return r;
}

#endif

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half __clc_lgamma_r(half x, private int *iptr) {
  return (half)__clc_lgamma_r((float)x, iptr);
}

#endif

#define __CLC_FUNCTION __clc_lgamma_r
#define __CLC_ARG2_TYPE int
#define __CLC_ADDRSPACE private
#define __CLC_BODY <clc/shared/unary_def_with_ptr_scalarize.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE
#undef __CLC_ARG2_TYPE
#undef __CLC_FUNCTION

#define __CLC_ADDRSPACE global
#define __CLC_BODY <clc_lgamma_r.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE

#define __CLC_ADDRSPACE local
#define __CLC_BODY <clc_lgamma_r.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE

#if _CLC_DISTINCT_GENERIC_AS_SUPPORTED
#define __CLC_ADDRSPACE generic
#define __CLC_BODY <clc_lgamma_r.inc>
#include <clc/math/gentype.inc>
#undef __CLC_ADDRSPACE
#endif
