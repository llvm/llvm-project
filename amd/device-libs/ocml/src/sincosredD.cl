/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

INLINEATTR double
MATH_PRIVATE(sincosred)(double x, __private double *cp)
{
    const double S0 = -0x1.5555555555555p-3;
    const double S1 =  0x1.1111111110bb3p-7;
    const double S2 = -0x1.a01a019e83e5cp-13;
    const double S3 =  0x1.71de3796cde01p-19;
    const double S4 = -0x1.ae600b42fdfa7p-26;
    const double S5 =  0x1.5e0b2f9a43bb8p-33;

    const double C0 =  0x1.5555555555555p-5;
    const double C1 = -0x1.6c16c16c16967p-10;
    const double C2 =  0x1.a01a019f4ec90p-16;
    const double C3 = -0x1.27e4fa17f65f6p-22;
    const double C4 =  0x1.1eeb69037ab78p-29;
    const double C5 = -0x1.907db46cc5e42p-37;

    double x2 = x*x;
    double r = 0.5 * x2;
    double t = 1.0 - r;
    double u = 1.0 - t;
    double v = u - r;

    double cx = t + MATH_MAD(x2*x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, C5, C4), C3), C2), C1), C0), v);
    double sx = MATH_MAD(x2*x, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, S5, S4), S3), S2), S1), S0), x);

    *cp = cx;
    return sx;
}

