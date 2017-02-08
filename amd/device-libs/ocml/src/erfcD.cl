/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

// Partially based on ideas from the Sun implementation
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

PUREATTR double
MATH_MANGLE(erfc)(double x)
{
    double ret;

    if (x < 0x1.e861fbb24c00ap-2) {
        if (x > -1.0) {
            double t = x * x;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      -0x1.abae491c443a9p-31, 0x1.d71b0f1b10a64p-27), -0x1.5c0726f04dcfbp-23), 0x1.b97fd3d992938p-20),
                      -0x1.f4ca4d6f3e30fp-17), 0x1.f9a2baa8fedd2p-14), -0x1.c02db03dd71d4p-11), 0x1.565bccf92b2f9p-8),
                      -0x1.b82ce311fa93ep-6), 0x1.ce2f21a040d16p-4), -0x1.812746b0379bdp-2), 0x1.20dd750429b6dp+0);
            ret = MATH_MAD(-x, ret, 1.0);
        } else if (x > -1.75) {
            double t = -x - 1.0;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      -0x1.6c922ed03eb9dp-17, 0x1.97d42571bbb38p-14), -0x1.41761e0138c87p-12), 0x1.7f635425509dep-13),
                      0x1.30fe6b148c32fp-10), -0x1.e682366d34981p-10), -0x1.39b7dcc1aeec8p-8), 0x1.f0ab5db978c52p-7),
                      0x1.2e3e92d3304b4p-8), -0x1.1b613d8e18405p-4), 0x1.1b614a01845b4p-4), 0x1.1b614b15ab5c1p-3),
                      -0x1.a911f0970fc8dp-2), 0x1.a911f096fbf43p-2), 0x1.d7bb3d3a08445p+0);
        } else if (x > -2.5) {
            double t = -x - 1.75;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      0x1.1f145e2e90ae8p-18, -0x1.04595429d0b58p-15), 0x1.566284cadc629p-14), -0x1.daefe4f2fa8e2p-17),
                      -0x1.cbee5eda62503p-12), 0x1.d416c2aa2275ap-11), 0x1.7eeb86b197684p-11), -0x1.8d11b66138741p-8),
                      0x1.25b37e361d1c9p-7), 0x1.b22258f45515dp-8), -0x1.8a0da54b7e9dep-5), 0x1.7148c3d5d2293p-4),
                      -0x1.7a4a8a2bdfeb2p-4), 0x1.b05530322115bp-5), 0x1.fc9683bfc6ab7p+0);
        } else if (x > -4.0) {
            double t = -x - 2.5;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      -0x1.708f6d0e65c33p-32, 0x1.dbd0618847c60p-28), -0x1.c3001cf83cd69p-26), -0x1.4dca746dfe625p-22),
                      0x1.a8e79a95d6f67p-20), 0x1.8d8d7711fc864p-16), -0x1.99fe2d9d9b69bp-13), -0x1.b3b1f1e28669cp-12),
                      0x1.01d3d83753fb1p-7), -0x1.e842cf8341e6ap-10), -0x1.a49bb4ab1d7d9p-3), 0x1.3a50e1b16e339p-1);
            ret = ret*ret;
            ret = ret*ret;
            ret = ret*ret;
            ret = MATH_MAD(-ret, ret, 2.0);
        } else if (x > -5.9375) {
            double t = -x - 4.0;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, 
                      0x1.5b22d2cd54932p-26, -0x1.3e056a1040a29p-24), -0x1.2d8f6bf8af04ap-19), 0x1.4c20d337a4541p-16),
                      0x1.d9d0971c8f96dp-16), -0x1.0a33e01adb0ddp-10), 0x1.63716fb40eab9p-9), 0x1.7d6f6bbcfc7e0p-6),
                      -0x1.5687476feec74p-3), 0x1.4cb2bacd30820p-2);
            ret = ret*ret;
            ret = ret*ret;
            ret = ret*ret;
            ret = MATH_MAD(-ret, ret, 2.0);
        } else {
            ret = 2.0;
        }
    } else {
        if (x < 1.0) {
            double t = x - 0.75;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, 
                      -0x1.57d59f658aba7p-16, 0x1.362e0b222318ep-14), 0x1.bc4dcd34fdd6dp-14), -0x1.470d403e0efe6p-11),
                      -0x1.86196ce26e31fp-13), 0x1.0410341ee1473p-8), -0x1.2db338db4ad88p-9), -0x1.2e0afac283b7fp-6),
                      0x1.b847796a479d8p-6), 0x1.b42a1890465d3p-5), -0x1.349b5eaa155b6p-3), -0x1.b6e8591f65270p-6),
                      0x1.edc5644353c2dp-2), -0x1.492e42d78d2c5p-1), 0x1.27c6d14c5e341p-2);
        } else if (x < 1.5) {
            double t = x - 1.25;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.9c25dae26e5a8p-18, 0x1.692456873fac4p-19), -0x1.d3ef7e77785bap-15), 0x1.baaa993d5590fp-15),
                      0x1.53b075bbc5b61p-12), -0x1.a00787b6af397p-11), -0x1.cc224fab0d8a4p-11), 0x1.75672d1e80999p-8),
                      -0x1.db43c97b37ceap-9), -0x1.5d0003afa1e92p-6), 0x1.8281ce0b36c0dp-5), 0x1.93a9a7bb80513p-8),
                      -0x1.571d01c5c56c8p-3), 0x1.2ebf3dcc9f22fp-2), -0x1.e4652fadcb6b2p-3), 0x1.3bcd133aa0ffcp-4);
        } else if (x < 1.75) {
            double t = x - 1.625;
            ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                  MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      0x1.02ad00dd8cbb4p-13, 0x1.70ffb4c1c5cbfp-12), -0x1.71c6788c68de8p-10), 0x1.2e4d6f91e46c7p-11),
                      0x1.954aa9df71457p-8), -0x1.d857f3fbcac79p-7), 0x1.17d430d63aaf5p-9), 0x1.974c0368aecfcp-5),
                      -0x1.d6631e1a2977fp-4), 0x1.0bcfca219477bp-3), -0x1.499d478bca733p-4), 0x1.612d893085125p-6);
        } else if (x < 27.21875) {
            double t = MATH_RCP(x*x);

            if (x < 2.75)
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                          0x1.ee796b0cccbebp+11, -0x1.f287322c462d4p+13), 0x1.d9e0700d3d82dp+14), -0x1.1a96768b6b29fp+15),
                          0x1.dafa2508a60dcp+14), -0x1.2bbd8e3460b89p+14), 0x1.27fd8cab24e6ep+13), -0x1.d7a7a4e4c3b93p+11),
                          0x1.37a4a4d018456p+10), -0x1.60173b9f73257p+8), 0x1.6253e7ca4b16fp+6), -0x1.51d02c514c31cp+4),
                          0x1.4e9a1546b2716p+2), -0x1.86ed776e3a5e5p+0), 0x1.3fb9e1ef8c40ap-1), -0x1.fffcb9ff22596p-2),
                          -0x1.43424dfcdbdcep-7);
            else
                ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                      MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, 
                          0x1.bba05f5648454p+38, -0x1.401ff919f9865p+39), 0x1.b23350c3b39a1p+38), -0x1.70d6cf6eca08ep+37),
                          0x1.b9e665656eee6p+35), -0x1.8f73b118a9b93p+33), 0x1.1da829fcea796p+31), -0x1.5090992846e0ep+28),
                          0x1.548adac0440f5p+25), -0x1.3694e9079941ep+22), 0x1.0e5ce4af6bb84p+19), -0x1.dda4fee0ea545p+15),
                          0x1.c3f3a46f6fac8p+12), -0x1.dc5f4d89f0ae7p+9), 0x1.1f825da9dcbacp+7), -0x1.98193f7900492p+4),
                          0x1.60fffd6b1743dp+2), -0x1.8aaaaa9e2e8dep+0), 0x1.3fffffffedba9p-1), -0x1.fffffffffff1fp-2),
                          -0x1.4341239e86f47p-7);

            double xh = AS_DOUBLE(AS_LONG(x) & 0xffffffff00000000L);
            ret = MATH_DIV(MATH_MANGLE(exp)(MATH_MAD(x - xh,  -(x + xh), ret)), x) *
                  MATH_MANGLE(exp)(MATH_MAD(xh, -xh, -0.5625));
        } else {
            ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN) ? x : 0.0;
        }
    }

    return ret;
}

