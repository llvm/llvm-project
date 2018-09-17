/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

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


struct ret_t {
    double result;
    int signp;
};

static struct ret_t
MATH_MANGLE(lgamma_r_impl)(double x)
{
    const double two52=  4.50359962737049600000e+15;
    const double pi  =  3.14159265358979311600e+00;
    const double a0  =  7.72156649015328655494e-02;
    const double a1  =  3.22467033424113591611e-01;
    const double a2  =  6.73523010531292681824e-02;
    const double a3  =  2.05808084325167332806e-02;
    const double a4  =  7.38555086081402883957e-03;
    const double a5  =  2.89051383673415629091e-03;
    const double a6  =  1.19270763183362067845e-03;
    const double a7  =  5.10069792153511336608e-04;
    const double a8  =  2.20862790713908385557e-04;
    const double a9  =  1.08011567247583939954e-04;
    const double a10 =  2.52144565451257326939e-05;
    const double a11 =  4.48640949618915160150e-05;
    const double tc  =  1.46163214496836224576e+00;
    const double tf  = -1.21486290535849611461e-01;
    const double tt  = -3.63867699703950536541e-18;
    const double t0  =  4.83836122723810047042e-01;
    const double t1  = -1.47587722994593911752e-01;
    const double t2  =  6.46249402391333854778e-02;
    const double t3  = -3.27885410759859649565e-02;
    const double t4  =  1.79706750811820387126e-02;
    const double t5  = -1.03142241298341437450e-02;
    const double t6  =  6.10053870246291332635e-03;
    const double t7  = -3.68452016781138256760e-03;
    const double t8  =  2.25964780900612472250e-03;
    const double t9  = -1.40346469989232843813e-03;
    const double t10 =  8.81081882437654011382e-04;
    const double t11 = -5.38595305356740546715e-04;
    const double t12 =  3.15632070903625950361e-04;
    const double t13 = -3.12754168375120860518e-04;
    const double t14 =  3.35529192635519073543e-04;
    const double u0  = -7.72156649015328655494e-02;
    const double u1  =  6.32827064025093366517e-01;
    const double u2  =  1.45492250137234768737e+00;
    const double u3  =  9.77717527963372745603e-01;
    const double u4  =  2.28963728064692451092e-01;
    const double u5  =  1.33810918536787660377e-02;
    const double v1  =  2.45597793713041134822e+00;
    const double v2  =  2.12848976379893395361e+00;
    const double v3  =  7.69285150456672783825e-01;
    const double v4  =  1.04222645593369134254e-01;
    const double v5  =  3.21709242282423911810e-03;
    const double s0  = -7.72156649015328655494e-02;
    const double s1  =  2.14982415960608852501e-01;
    const double s2  =  3.25778796408930981787e-01;
    const double s3  =  1.46350472652464452805e-01;
    const double s4  =  2.66422703033638609560e-02;
    const double s5  =  1.84028451407337715652e-03;
    const double s6  =  3.19475326584100867617e-05;
    const double r1  =  1.39200533467621045958e+00;
    const double r2  =  7.21935547567138069525e-01;
    const double r3  =  1.71933865632803078993e-01;
    const double r4  =  1.86459191715652901344e-02;
    const double r5  =  7.77942496381893596434e-04;
    const double r6  =  7.32668430744625636189e-06;
    const double w0  =  4.18938533204672725052e-01;
    const double w1  =  8.33333333333329678849e-02;
    const double w2  = -2.77777777728775536470e-03;
    const double w3  =  7.93650558643019558500e-04;
    const double w4  = -5.95187557450339963135e-04;
    const double w5  =  8.36339918996282139126e-04;
    const double w6  = -1.63092934096575273989e-03;
    const double z1  = -0x1.2788cfc6fb619p-1;
    const double z2  =  0x1.a51a6625307d3p-1;
    const double z3  = -0x1.9a4d55beab2d7p-2;
    const double z4  =  0x1.151322ac7d848p-2;
    const double z5  = -0x1.a8b9c17aa6149p-3;

    double ax = BUILTIN_ABS_F64(x);
    uint hax = AS_UINT2(ax).hi;
    double ret;

    if (hax < 0x3f700000) {
        // ax < 0x1.0p-8
        ret = MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, z5, z4), z3), z2), z1),
                       -MATH_MANGLE(log)(ax));
    } else if (hax < 0x40000000) {
        // ax < 2.0
        int i;
        bool c;
        double y, t;
        if (hax <= 0x3feccccc) { // |x| < 0.9 : lgamma(x) = lgamma(x+1)-log(x)
            ret = -MATH_MANGLE(log)(ax);

            y = 1.0 - ax;
            i = 0;

            c = hax < 0x3FE76944; // x < 0.7316
            t = ax - (tc - 1.0);
            y = c ? t : y;
            i = c ? 1 : i;

            c = hax < 0x3FCDA661; // x < .2316
            y = c ? ax : y;
            i = c ? 2 : i;
        } else {
            ret = 0.0;

            y = 2.0 - ax;
            i = 0;

            c = hax < 0x3FFBB4C3; // x < 1.7316
            t = ax - tc;
            y = c ? t : y;
            i = c ? 1 : i;

            c = hax < 0x3FF3B4C4; // x < 1.2316
            t = ax - 1.0;
            y = c ? t : y;
            i = c ? 2 : i;
        }

        double w, z, p, p1, p2, p3;
        switch(i) {
        case 0:
            z = y*y;
            p1 = MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, a10, a8), a6), a4), a2), a0);
            p2 = z * MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, MATH_MAD(z, a11, a9), a7), a5), a3), a1);
            p = MATH_MAD(y, p1, p2);
            ret += MATH_MAD(y, -0.5, p);
            break;
        case 1:
            z = y*y;
            w = z*y;
            p1 = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, t12, t9), t6), t3), t0);
            p2 = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, t13, t10), t7), t4), t1);
            p3 = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, t14, t11), t8), t5), t2);
            p = MATH_MAD(z, p1, -MATH_MAD(w, -MATH_MAD(y, p3,p2), tt));
            ret += tf + p;
            break;
        case 2:
            p1 = y * MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, u5, u4), u3), u2), u1), u0);
            p2 = MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, v5, v4), v3), v2), v1), 1.0);
            ret += MATH_MAD(y, -0.5, MATH_DIV(p1, p2));
            break;
        }
    } else if (hax < 0x40200000) { // 2 < ax < 8
        int i = (int)ax;
        double y = ax - (double)i;
        double p = y * MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, s6, s5), s4), s3), s2), s1), s0);
        double q = MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, r6, r5), r4), r3), r2), r1), 1.0);
        ret = MATH_MAD(y, 0.5, MATH_DIV(p, q));

        double y2 = y + 2.0;
        double y3 = y + 3.0;
        double y4 = y + 4.0;
        double y5 = y + 5.0;
        double y6 = y + 6.0;

        double z = 1.0;
        z *= i > 2 ? y2 : 1.0;
        z *= i > 3 ? y3 : 1.0;
        z *= i > 4 ? y4 : 1.0;
        z *= i > 5 ? y5 : 1.0;
        z *= i > 6 ? y6 : 1.0;

        ret += MATH_MANGLE(log)(z);
    } else if (hax < 0x43900000) { // 8 <= ax < 2^58
        double z = MATH_RCP(ax);
        double y = z*z;
        double w = MATH_MAD(z, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, w6, w5), w4), w3), w2), w1), w0);
        ret = MATH_MAD(ax - 0.5, MATH_MANGLE(log)(ax) - 1.0, w);
    } else  { // 2^58 <= ax <= Inf
        ret = MATH_MAD(ax, MATH_MANGLE(log)(ax), -ax);
    }


    int s = 0;
    if (x >= 0.0) {
        ret = x == 1.0 | x == 2.0 ? 0.0 : ret;
        s = x == 0.0 ? 0 : 1;
    } else if (hax < 0x43300000) { // x > -0x1.0p+52
        double t = MATH_MANGLE(sinpi)(x);
        double negadj = MATH_MANGLE(log)(MATH_DIV(pi, BUILTIN_ABS_F64(t * x)));
        ret = negadj - ret;
        bool z = BUILTIN_FRACTION_F64(x) == 0.0;
        ret = z ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        s = t < 0.0 ? -1 : 1;
        s = z ? 0 : s;
    }

    if (!FINITE_ONLY_OPT()) {
        // Handle negative integer, Inf, NaN
        ret = BUILTIN_CLASS_F64(ax, CLASS_NZER|CLASS_PZER|CLASS_PINF) | (x < 0.0f & hax >= 0x43300000) ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = BUILTIN_ISNAN_F64(x) ? x : ret;
    }

    struct ret_t result;
    result.result = ret;
    result.signp = s;
    return result;
}


double
MATH_MANGLE(lgamma_r)(double x, __private int *signp)
{
    struct ret_t ret = MATH_MANGLE(lgamma_r_impl)(x);
    *signp = ret.signp;
    return ret.result;
}
