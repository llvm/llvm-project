/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

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
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 1.25) {
        if (ax >= 0.84375) { // .84375 <= |x| < 1.25
            double s = ax - 1.0;

            double P = MATH_MAD(s,
                           MATH_MAD(s,
                               MATH_MAD(s,
                                   MATH_MAD(s,
                                       MATH_MAD(s,
                                           MATH_MAD(s, -0x1.1bf380a96073fp-9, 0x1.22a36599795ebp-5),
                                           -0x1.c63983d3e28ecp-4),
                                       0x1.45fca805120e4p-2),
                                   -0x1.7d240fbb8c3f1p-2),
                               0x1.a8d00ad92b34dp-2),
                           -0x1.359b8bef77538p-9); 

            double Q = MATH_MAD(s,
                           MATH_MAD(s,
                               MATH_MAD(s,
                                   MATH_MAD(s,
                                       MATH_MAD(s,
                                           MATH_MAD(s, 0x1.88b545735151dp-7, 0x1.bedc26b51dd1cp-7),
                                           0x1.02660e763351fp-3),
                                       0x1.2635cd99fe9a7p-4),
                                   0x1.14af092eb6f33p-1),
                               0x1.b3e6618eee323p-4),
                           1.0);

            double pbyq = MATH_DIV(P,Q);
            const double erx = 8.45062911510467529297e-01;
            double retn = erx + pbyq + 1.0;
            double retp = 1.0 - erx - pbyq;
            ret = x < 0.0 ? retn : retp;
        } else if (ax >= 0x1.0p-56) {
            double z = x * x;

            double r = MATH_MAD(z,
                           MATH_MAD(z,
                               MATH_MAD(z,
                                   MATH_MAD(z, -0x1.8ead6120016acp-16, -0x1.7a291236668e4p-8),
                                   -0x1.d2a51dbd7194fp-6),
                               -0x1.4cd7d691cb913p-2),
                           0x1.06eba8214db68p-3);

            double s = MATH_MAD(z,
                           MATH_MAD(z,
                               MATH_MAD(z,
                                   MATH_MAD(z,
                                       MATH_MAD(z, -0x1.09c4342a26120p-18, 0x1.15dc9221c1a10p-13),
                                       0x1.4d022c4d36b0fp-8),
                                   0x1.0a54c5536cebap-4),
                               0x1.97779cddadc09p-2),
                           1.0);

            double y = MATH_DIV(r , s);

            double retl = 1.0 - MATH_MAD(x, y, x);
            double retg = 0.5 - MATH_MAD(x, y, x - 0.5);
            ret = x < 0.25 ? retl : retg;
        } else { // |x| < 2**-56
            ret = 1.0 - x; // In fact, this should be 1.0
        }
    } else if (x >= -6.0 && x < 27.23)  {
        double s = MATH_DIV(1.0 , ax*ax);
        double R, S;

        if (ax < 2.8571428571428571428571428571429) { // |x| < 1/.35
            R = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s,
                                        MATH_MAD(s, -0x1.3a0efc69ac25cp+3, -0x1.4526557e4d2f2p+6),
                                        -0x1.7135cebccabb2p+7),
                                    -0x1.44cb184282266p+7),
                                -0x1.f300ae4cba38dp+5),
                            -0x1.51e0441b0e726p+3),
                        -0x1.63416e4ba7360p-1),
                    -0x1.43412600d6435p-7);

            S = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s,
                                        MATH_MAD(s,
                                            MATH_MAD(s, -0x1.eeff2ee749a62p-5, 0x1.a47ef8e484a93p+2),
                                            0x1.b28a3ee48ae2cp+6),
                                        0x1.ad02157700314p+8),
                                    0x1.42b1921ec2868p+9),
                                0x1.b290dd58a1a71p+8),
                            0x1.1350c526ae721p+7),
                        0x1.3a6b9bd707687p+4),
                    1.0);
        } else { // |x| >= 1/.35
            R = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s, -0x1.e384e9bdc383fp+8, -0x1.004616a2e5992p+10),
                                    -0x1.3ec881375f228p+9),
                                -0x1.4145d43c5ed98p+7),
                            -0x1.1c209555f995ap+4),
                        -0x1.993ba70c285dep-1),
                    -0x1.4341239e86f4ap-7);

            S = MATH_MAD(s,
                    MATH_MAD(s,
                        MATH_MAD(s,
                            MATH_MAD(s,
                                MATH_MAD(s,
                                    MATH_MAD(s,
                                        MATH_MAD(s, -0x1.670e242712d62p+4, 0x1.da874e79fe763p+8),
                                        0x1.3f219cedf3be6p+11),
                                    0x1.8ffb7688c246ap+11),
                                0x1.802eb189d5118p+10),
                            0x1.45cae221b9f0ap+8),
                        0x1.e568b261d5190p+4),
                    1.0);
        }
        
        double z = AS_DOUBLE(AS_LONG(ax) & 0xfffffffff8000000L);
        double r = MATH_MANGLE(exp)(MATH_MAD(z, -z, -0.5625)) *
                   MATH_MANGLE(exp)(MATH_MAD(z - ax,  z + ax, MATH_DIV(R , S)));
        r = MATH_DIV(r, ax);
        double retn = 2.0 - r;
        ret = x < 0.0 ? retn : r;
    } else if (x < -6.0) {
        ret = 2.0;
    } else if (ax > 27.23) {
        ret = 0.0;
    }

    ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN) ? x : ret;
    return ret;
}

