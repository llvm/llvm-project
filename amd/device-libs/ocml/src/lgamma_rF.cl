/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

// This lgamma routine began with Sun's lgamma code from netlib.
// Their original copyright notice follows.
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

/* Reentrant version of the logarithm of the Gamma function
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

struct ret_t {
    float result;
    int signp;
};

static struct ret_t
MATH_MANGLE(lgamma_r_impl)(float x)
{
    const float two52 =  4.50359962737049600000e+15f;
    const float pi  =  3.14159265358979311600e+00f;
    const float a0  =  7.72156649015328655494e-02f;
    const float a1  =  3.22467033424113591611e-01f;
    const float a2  =  6.73523010531292681824e-02f;
    const float a3  =  2.05808084325167332806e-02f;
    const float a4  =  7.38555086081402883957e-03f;
    const float a5  =  2.89051383673415629091e-03f;
    const float a6  =  1.19270763183362067845e-03f;
    const float a7  =  5.10069792153511336608e-04f;
    const float a8  =  2.20862790713908385557e-04f;
    const float a9  =  1.08011567247583939954e-04f;
    const float a10 =  2.52144565451257326939e-05f;
    const float a11 =  4.48640949618915160150e-05f;
    const float tc  =  1.46163214496836224576e+00f;
    const float tf  = -1.21486290535849611461e-01f;
    const float tt  = -3.63867699703950536541e-18f;
    const float t0  =  4.83836122723810047042e-01f;
    const float t1  = -1.47587722994593911752e-01f;
    const float t2  =  6.46249402391333854778e-02f;
    const float t3  = -3.27885410759859649565e-02f;
    const float t4  =  1.79706750811820387126e-02f;
    const float t5  = -1.03142241298341437450e-02f;
    const float t6  =  6.10053870246291332635e-03f;
    const float t7  = -3.68452016781138256760e-03f;
    const float t8  =  2.25964780900612472250e-03f;
    const float t9  = -1.40346469989232843813e-03f;
    const float t10 =  8.81081882437654011382e-04f;
    const float t11 = -5.38595305356740546715e-04f;
    const float t12 =  3.15632070903625950361e-04f;
    const float t13 = -3.12754168375120860518e-04f;
    const float t14 =  3.35529192635519073543e-04f;
    const float u0  = -7.72156649015328655494e-02f;
    const float u1  =  6.32827064025093366517e-01f;
    const float u2  =  1.45492250137234768737e+00f;
    const float u3  =  9.77717527963372745603e-01f;
    const float u4  =  2.28963728064692451092e-01f;
    const float u5  =  1.33810918536787660377e-02f;
    const float v1  =  2.45597793713041134822e+00f;
    const float v2  =  2.12848976379893395361e+00f;
    const float v3  =  7.69285150456672783825e-01f;
    const float v4  =  1.04222645593369134254e-01f;
    const float v5  =  3.21709242282423911810e-03f;
    const float s0  = -7.72156649015328655494e-02f;
    const float s1  =  2.14982415960608852501e-01f;
    const float s2  =  3.25778796408930981787e-01f;
    const float s3  =  1.46350472652464452805e-01f;
    const float s4  =  2.66422703033638609560e-02f;
    const float s5  =  1.84028451407337715652e-03f;
    const float s6  =  3.19475326584100867617e-05f;
    const float r1  =  1.39200533467621045958e+00f;
    const float r2  =  7.21935547567138069525e-01f;
    const float r3  =  1.71933865632803078993e-01f;
    const float r4  =  1.86459191715652901344e-02f;
    const float r5  =  7.77942496381893596434e-04f;
    const float r6  =  7.32668430744625636189e-06f;
    const float w0  =  4.18938533204672725052e-01f;
    const float w1  =  8.33333333333329678849e-02f;
    const float w2  = -2.77777777728775536470e-03f;
    const float w3  =  7.93650558643019558500e-04f;
    const float w4  = -5.95187557450339963135e-04f;
    const float w5  =  8.36339918996282139126e-04f;
    const float w6  = -1.63092934096575273989e-03f;
    const float z1  = -0x1.2788d0p-1f;
    const float z2  =  0x1.a51a66p-1f;
    const float z3  = -0x1.9a4d56p-2f;
    const float z4  =  0x1.151322p-2f;

    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 0x1.0p-6f) {
        ret = MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, z4, z3), z2), z1),
                       -MATH_MANGLE(log)(ax));
    } else if (ax < 2.0f) {
        int i;
        bool c;
        float y, t;
        if( ax <= 0.9f) { // lgamma(x) = lgamma(x+1)-log(x)
            ret = -MATH_MANGLE(log)(ax);
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

        float z, w, p1, p2, p3, p;
        switch(i) {
        case 0:
            z = y * y;
            p1 = MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, a10, a8), a6), a4), a2), a0);
            p2 = z * MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, a11, a9), a7), a5), a3), a1);
            p = MATH_MAD(y, p1, p2);
            ret += MATH_MAD(y, -0.5f, p);
            break;
        case 1:
            z = y * y;
            w = z * y;
            p1 = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, t12, t9), t6), t3), t0);
            p2 = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, t13, t10), t7), t4), t1);
            p3 = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, t14, t11), t8), t5), t2);
            p = MATH_MAD(z, p1, -MATH_MAD(w, -MATH_MAD(y, p3, p2), tt));
            ret += tf + p;
            break;
        case 2:
            p1 = y * MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, u5, u4), u3), u2), u1), u0);
            p2 = MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, v5, v4), v3), v2), v1), 1.0f);
            ret += MATH_MAD(y, -0.5f, MATH_FAST_DIV(p1, p2));
            break;
        }
    } else if (ax < 8.0f) {  // 2 < |x| < 8
        int i = (int)ax;
        float y = ax - (float) i;
        float p = y * MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, s6, s5), s4), s3), s2), s1), s0);
        float q = MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, r6, r5), r4), r3), r2), r1), 1.0f);
        ret = MATH_MAD(y, 0.5f, MATH_FAST_DIV(p, q));

        float y2 = y + 2.0f;
        float y3 = y + 3.0f;
        float y4 = y + 4.0f;
        float y5 = y + 5.0f;
        float y6 = y + 6.0f;

        float z = 1.0f;
        z *= i > 2 ? y2 : 1.0f;
        z *= i > 3 ? y3 : 1.0f;
        z *= i > 4 ? y4 : 1.0f;
        z *= i > 5 ? y5 : 1.0f;
        z *= i > 6 ? y6 : 1.0f;

        ret += MATH_MANGLE(log)(z);
    } else if (ax < 0x1.0p+58f) { // 8 <= |x| < 2^58
        float z = MATH_FAST_RCP(ax);
        float y = z * z;
        float w = MATH_MAD(z, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, w6, w5), w4), w3), w2), w1), w0);
        ret = MATH_MAD(ax - 0.5f, MATH_MANGLE(log)(ax) - 1.0f, w);
    } else {
        // 2^58 <= |x| <= Inf
        ret = MATH_MAD(ax, MATH_MANGLE(log)(ax), -ax);
    }

    int s = 0;
    if (x >= 0.0f) {
        ret = ((x == 1.0f) | (x == 2.0f)) ? 0.0f : ret;
        s = x == 0.0f ? 0 : 1;
    } else if (ax < 0x1.0p+23f) { // x > -0x1.0p+23
        if (ax > 0x1.0p-21f) {
            float t = MATH_MANGLE(sinpi)(x);
            float negadj = MATH_MANGLE(log)(MATH_DIV(pi, BUILTIN_ABS_F32(t * x)));
            ret = negadj - ret;
            bool z = BUILTIN_FRACTION_F32(x) == 0.0f;
            ret = z ? AS_FLOAT(PINFBITPATT_SP32) : ret;
            s = t < 0.0f ? -1 : 1;
            s = z ? 0 : s;
        } else {
            s = -1;
        }
    }

    if (!FINITE_ONLY_OPT()) {
        ret = ((ax != 0.0f) && !BUILTIN_ISINF_F32(ax) &&
              ((x >= 0.0f) || (ax < 0x1.0p+23f))) ? ret : AS_FLOAT(PINFBITPATT_SP32);

        ret = BUILTIN_ISNAN_F32(x) ? x : ret;
    }

    struct ret_t result;
    result.result = ret;
    result.signp = s;

    return result;
}

float
MATH_MANGLE(lgamma_r)(float x, __private int *signp)
{
    struct ret_t ret = MATH_MANGLE(lgamma_r_impl)(x);
    *signp = ret.signp;
    return ret.result;
}
