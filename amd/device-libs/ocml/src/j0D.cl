/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern double MATH_PRIVATE(sincosb)(double, int, __private double *);
extern double MATH_PRIVATE(pzero)(double);
extern double MATH_PRIVATE(qzero)(double);

// This implementation makes use of large x approximations from
// the Sun library which reqires the following to be included:
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

double
MATH_MANGLE(j0)(double x)
{
    x = BUILTIN_ABS_F64(x);

    const double b0 = 1.65625;
    const double b1 = 3.125;
    const double b2 = 4.6875;
    const double b3 = 6.265625;
    const double b4 = 7.84375;
    const double b5 = 9.421875;
    const double b6 = 10.984375;
    const double b7 = 12.578125;

    double ret;

    if (x <= b7) {
        // Ty to maintain relative accuracy here

        USE_TABLE(double, p, M64_J0);
        double ch, cl;

        if (x <= b3) {
            if (x <= b0) {
                ch = 0.0;
                cl = 0.0;
            } else if (x <= b1) {
                ch = 0x1.33d152e971b40p+1;
                cl = -0x1.0f539d7da258ep-53;
                p += 1*15;
            } else if (x <= b2) {
                ch = 0x1.ea75575af6f09p+1;
                cl = -0x1.60155a9d1b256p-53;
                p += 2*15;
            } else {
                ch = 0x1.6148f5b2c2e45p+2;
                cl = 0x1.75054cd60a517p-54;
                p += 3*15;
            }
        } else {
            if (x <= b4) {
                ch = 0x1.c0ff5f3b47250p+2;
                cl = -0x1.b226d9d243827p-54;
                p += 4*15;
            } else if (x <= b5) {
                ch = 0x1.14eb56cccdecap+3;
                cl = -0x1.51970714c7c25p-52;
                p += 5*15;
            } else if (x <= b6) {
                ch = 0x1.458d0d0bdfc29p+3;
                cl = 0x1.02610a51562b6p-51;
                p += 6*15;
            } else {
                ch = 0x1.79544008272b6p+3;
                cl = 0x1.444fd5821d5b1p-52;
                p += 7*15;
            }
        }

        x = x - ch - cl;
        ret = MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
              MATH_MAD(x, MATH_MAD(x,
              p[14], p[13]), p[12]),
              p[11]), p[10]), p[9]), p[8]),
              p[7]), p[6]), p[5]), p[4]),
              p[3]), p[2]), p[1]), p[0]);
              
    } else {
        // j0(x) ~ sqrt(2 / (pi*x)) * (P0(x) cos(x-pi/4) - Q0(x) sin(x-pi/4))
        double c;
        double s = MATH_PRIVATE(sincosb)(x, 0, &c);
        const double sqrt2bypi = 0x1.9884533d43651p-1;
        if (x > 0x1.0p+129)
            ret = MATH_DIV(sqrt2bypi * c, MATH_SQRT(x));
        else
            ret = MATH_DIV(sqrt2bypi * (MATH_PRIVATE(pzero)(x)*c - MATH_PRIVATE(qzero)(x)*s), MATH_SQRT(x));
        ret = BUILTIN_CLASS_F64(x, CLASS_PINF) ? 0.0 : ret;
    }

    return ret;
}

