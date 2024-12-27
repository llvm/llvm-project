/*
 * Copyright (c) 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include <clc/clc.h>
#include <clc/clcmacro.h>

#include "math.h"

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

#define erx   8.4506291151e-01f        /* 0x3f58560b */

// Coefficients for approximation to  erf on [00.84375]

#define efx   1.2837916613e-01f        /* 0x3e0375d4 */
#define efx8  1.0270333290e+00f        /* 0x3f8375d4 */

#define pp0   1.2837916613e-01f        /* 0x3e0375d4 */
#define pp1  -3.2504209876e-01f        /* 0xbea66beb */
#define pp2  -2.8481749818e-02f        /* 0xbce9528f */
#define pp3  -5.7702702470e-03f        /* 0xbbbd1489 */
#define pp4  -2.3763017452e-05f        /* 0xb7c756b1 */
#define qq1   3.9791721106e-01f        /* 0x3ecbbbce */
#define qq2   6.5022252500e-02f        /* 0x3d852a63 */
#define qq3   5.0813062117e-03f        /* 0x3ba68116 */
#define qq4   1.3249473704e-04f        /* 0x390aee49 */
#define qq5  -3.9602282413e-06f        /* 0xb684e21a */

// Coefficients for approximation to  erf  in [0.843751.25]

#define pa0  -2.3621185683e-03f        /* 0xbb1acdc6 */
#define pa1   4.1485610604e-01f        /* 0x3ed46805 */
#define pa2  -3.7220788002e-01f        /* 0xbebe9208 */
#define pa3   3.1834661961e-01f        /* 0x3ea2fe54 */
#define pa4  -1.1089469492e-01f        /* 0xbde31cc2 */
#define pa5   3.5478305072e-02f        /* 0x3d1151b3 */
#define pa6  -2.1663755178e-03f        /* 0xbb0df9c0 */
#define qa1   1.0642088205e-01f        /* 0x3dd9f331 */
#define qa2   5.4039794207e-01f        /* 0x3f0a5785 */
#define qa3   7.1828655899e-02f        /* 0x3d931ae7 */
#define qa4   1.2617121637e-01f        /* 0x3e013307 */
#define qa5   1.3637083583e-02f        /* 0x3c5f6e13 */
#define qa6   1.1984500103e-02f        /* 0x3c445aa3 */

// Coefficients for approximation to  erfc in [1.251/0.35]

#define ra0  -9.8649440333e-03f        /* 0xbc21a093 */
#define ra1  -6.9385856390e-01f        /* 0xbf31a0b7 */
#define ra2  -1.0558626175e+01f        /* 0xc128f022 */
#define ra3  -6.2375331879e+01f        /* 0xc2798057 */
#define ra4  -1.6239666748e+02f        /* 0xc322658c */
#define ra5  -1.8460508728e+02f        /* 0xc3389ae7 */
#define ra6  -8.1287437439e+01f        /* 0xc2a2932b */
#define ra7  -9.8143291473e+00f        /* 0xc11d077e */
#define sa1   1.9651271820e+01f        /* 0x419d35ce */
#define sa2   1.3765776062e+02f        /* 0x4309a863 */
#define sa3   4.3456588745e+02f        /* 0x43d9486f */
#define sa4   6.4538726807e+02f        /* 0x442158c9 */
#define sa5   4.2900814819e+02f        /* 0x43d6810b */
#define sa6   1.0863500214e+02f        /* 0x42d9451f */
#define sa7   6.5702495575e+00f        /* 0x40d23f7c */
#define sa8  -6.0424413532e-02f        /* 0xbd777f97 */

// Coefficients for approximation to  erfc in [1/.3528]

#define rb0  -9.8649431020e-03f        /* 0xbc21a092 */
#define rb1  -7.9928326607e-01f        /* 0xbf4c9dd4 */
#define rb2  -1.7757955551e+01f        /* 0xc18e104b */
#define rb3  -1.6063638306e+02f        /* 0xc320a2ea */
#define rb4  -6.3756646729e+02f        /* 0xc41f6441 */
#define rb5  -1.0250950928e+03f        /* 0xc480230b */
#define rb6  -4.8351919556e+02f        /* 0xc3f1c275 */
#define sb1   3.0338060379e+01f        /* 0x41f2b459 */
#define sb2   3.2579251099e+02f        /* 0x43a2e571 */
#define sb3   1.5367296143e+03f        /* 0x44c01759 */
#define sb4   3.1998581543e+03f        /* 0x4547fdbb */
#define sb5   2.5530502930e+03f        /* 0x451f90ce */
#define sb6   4.7452853394e+02f        /* 0x43ed43a7 */
#define sb7  -2.2440952301e+01f        /* 0xc1b38712 */

_CLC_OVERLOAD _CLC_DEF float erf(float x) {
    int hx = as_uint(x);
    int ix = hx & 0x7fffffff;
    float absx = as_float(ix);

    float x2 = absx * absx;
    float t = 1.0f / x2;
    float tt = absx - 1.0f;
    t = absx < 1.25f ? tt : t;
    t = absx < 0.84375f ? x2 : t;

    float u, v, tu, tv;

    // |x| < 6
    u = mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, rb6, rb5), rb4), rb3), rb2), rb1), rb0);
    v = mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, sb7, sb6), sb5), sb4), sb3), sb2), sb1);

    tu = mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, ra7, ra6), ra5), ra4), ra3), ra2), ra1), ra0);
    tv = mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, sa8, sa7), sa6), sa5), sa4), sa3), sa2), sa1);
    u = absx < 0x1.6db6dcp+1f ? tu : u;
    v = absx < 0x1.6db6dcp+1f ? tv : v;

    tu = mad(t, mad(t, mad(t, mad(t, mad(t, mad(t, pa6, pa5), pa4), pa3), pa2), pa1), pa0);
    tv = mad(t, mad(t, mad(t, mad(t, mad(t, qa6, qa5), qa4), qa3), qa2), qa1);
    u = absx < 1.25f ? tu : u;
    v = absx < 1.25f ? tv : v;

    tu = mad(t, mad(t, mad(t, mad(t, pp4, pp3), pp2), pp1), pp0);
    tv = mad(t, mad(t, mad(t, mad(t, qq5, qq4), qq3), qq2), qq1);
    u = absx < 0.84375f ? tu : u;
    v = absx < 0.84375f ? tv : v;

    v = mad(t, v, 1.0f);
    float q = MATH_DIVIDE(u, v);

    float ret = 1.0f;

    // |x| < 6
    float z = as_float(ix & 0xfffff000);
    float r = exp(mad(-z, z, -0.5625f)) * exp(mad(z-absx, z+absx, q));
    r = 1.0f - MATH_DIVIDE(r,  absx);
    ret = absx < 6.0f ? r : ret;

    r = erx + q;
    ret = absx < 1.25f ? r : ret;

    ret = as_float((hx & 0x80000000) | as_int(ret));

    r = mad(x, q, x);
    ret = absx < 0.84375f ? r : ret;

    // Prevent underflow
    r = 0.125f * mad(8.0f, x, efx8 * x);
    ret = absx < 0x1.0p-28f ? r : ret;

    ret = isnan(x) ? x : ret;

    return ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, float, erf, float);

#ifdef cl_khr_fp64

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

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

#define AU0 -9.86494292470009928597e-03
#define AU1 -7.99283237680523006574e-01
#define AU2 -1.77579549177547519889e+01
#define AU3 -1.60636384855821916062e+02
#define AU4 -6.37566443368389627722e+02
#define AU5 -1.02509513161107724954e+03
#define AU6 -4.83519191608651397019e+02

#define AV1  3.03380607434824582924e+01
#define AV2  3.25792512996573918826e+02
#define AV3  1.53672958608443695994e+03
#define AV4  3.19985821950859553908e+03
#define AV5  2.55305040643316442583e+03
#define AV6  4.74528541206955367215e+02
#define AV7 -2.24409524465858183362e+01

#define BU0 -9.86494403484714822705e-03
#define BU1 -6.93858572707181764372e-01
#define BU2 -1.05586262253232909814e+01
#define BU3 -6.23753324503260060396e+01
#define BU4 -1.62396669462573470355e+02
#define BU5 -1.84605092906711035994e+02
#define BU6 -8.12874355063065934246e+01
#define BU7 -9.81432934416914548592e+00

#define BV1  1.96512716674392571292e+01
#define BV2  1.37657754143519042600e+02
#define BV3  4.34565877475229228821e+02
#define BV4  6.45387271733267880336e+02
#define BV5  4.29008140027567833386e+02
#define BV6  1.08635005541779435134e+02
#define BV7  6.57024977031928170135e+00
#define BV8 -6.04244152148580987438e-02

#define CU0 -2.36211856075265944077e-03
#define CU1  4.14856118683748331666e-01
#define CU2 -3.72207876035701323847e-01
#define CU3  3.18346619901161753674e-01
#define CU4 -1.10894694282396677476e-01
#define CU5  3.54783043256182359371e-02
#define CU6 -2.16637559486879084300e-03

#define CV1  1.06420880400844228286e-01
#define CV2  5.40397917702171048937e-01
#define CV3  7.18286544141962662868e-02
#define CV4  1.26171219808761642112e-01
#define CV5  1.36370839120290507362e-02
#define CV6  1.19844998467991074170e-02

#define DU0  1.28379167095512558561e-01
#define DU1 -3.25042107247001499370e-01
#define DU2 -2.84817495755985104766e-02
#define DU3 -5.77027029648944159157e-03
#define DU4 -2.37630166566501626084e-05

#define DV1  3.97917223959155352819e-01
#define DV2  6.50222499887672944485e-02
#define DV3  5.08130628187576562776e-03
#define DV4  1.32494738004321644526e-04
#define DV5 -3.96022827877536812320e-06

_CLC_OVERLOAD _CLC_DEF double erf(double y) {
    double x = fabs(y);
    double x2 = x * x;
    double xm1 = x - 1.0;

    // Poly variable
    double t = 1.0 / x2;
    t = x < 1.25 ? xm1 : t;
    t = x < 0.84375 ? x2 : t;

    double u, ut, v, vt;

    // Evaluate rational poly
    // XXX We need to see of we can grab 16 coefficents from a table
    // faster than evaluating 3 of the poly pairs
    // if (x < 6.0)
    u = fma(t, fma(t, fma(t, fma(t, fma(t, fma(t, AU6, AU5), AU4), AU3), AU2), AU1), AU0);
    v = fma(t, fma(t, fma(t, fma(t, fma(t, fma(t, AV7, AV6), AV5), AV4), AV3), AV2), AV1);

    ut = fma(t, fma(t, fma(t, fma(t, fma(t, fma(t, fma(t, BU7, BU6), BU5), BU4), BU3), BU2), BU1), BU0);
    vt = fma(t, fma(t, fma(t, fma(t, fma(t, fma(t, fma(t, BV8, BV7), BV6), BV5), BV4), BV3), BV2), BV1);
    u = x < 0x1.6db6ep+1 ? ut : u;
    v = x < 0x1.6db6ep+1 ? vt : v;

    ut = fma(t, fma(t, fma(t, fma(t, fma(t, fma(t, CU6, CU5), CU4), CU3), CU2), CU1), CU0);
    vt = fma(t, fma(t, fma(t, fma(t, fma(t, CV6, CV5), CV4), CV3), CV2), CV1);
    u = x < 1.25 ? ut : u;
    v = x < 1.25 ? vt : v;

    ut = fma(t, fma(t, fma(t, fma(t, DU4, DU3), DU2), DU1), DU0);
    vt = fma(t, fma(t, fma(t, fma(t, DV5, DV4), DV3), DV2), DV1);
    u = x < 0.84375 ? ut : u;
    v = x < 0.84375 ? vt : v;

    v = fma(t, v, 1.0);

    // Compute rational approximation
    double q = u / v;

    // Compute results
    double z = as_double(as_long(x) & 0xffffffff00000000L);
    double r = exp(-z * z - 0.5625) * exp((z - x) * (z + x) + q);
    r = 1.0 - r / x;

    double ret = x < 6.0 ? r : 1.0;

    r = 8.45062911510467529297e-01 + q;
    ret = x < 1.25 ? r : ret;

    q = x < 0x1.0p-28 ? 1.28379167095512586316e-01 : q;

    r = fma(x, q, x);
    ret = x < 0.84375 ? r : ret;

    ret = isnan(x) ? x : ret;

    return y < 0.0 ? -ret : ret;
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, double, erf, double);

#ifdef cl_khr_fp16

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

_CLC_OVERLOAD _CLC_DEF half erf(half h) {
    return (half)erf((float)h);
}

_CLC_UNARY_VECTORIZE(_CLC_OVERLOAD _CLC_DEF, half, erf, half);

#endif

#endif
