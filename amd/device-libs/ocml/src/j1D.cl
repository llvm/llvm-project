
#include "mathD.h"

extern double MATH_PRIVATE(sincosb)(double, int, __private double *);
extern double MATH_PRIVATE(pone)(double);
extern double MATH_PRIVATE(qone)(double);


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
MATH_MANGLE(j1)(double x)
{
    const double b0 =  1.09375;
    const double b1 =  2.84375;
    const double b2 =  4.578125;
    const double b3 =  6.171875;
    const double b4 =  7.78125;
    const double b5 =  9.359375;
    const double b6 = 10.953125;
    const double b7 = 12.515625;

    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax <= b7) {
        // Ty to maintain relative accuracy here

        USE_TABLE(double, p, M64_J1);
        double ch, cl;

        if (ax <= b3) {
            if (ax <= b0) {
                ch = 0.0;
                cl = 0.0;
            } else if (ax <= b1) {
                ch = 0x1.d757d1fec8a3ap+0;
                cl = 0x1.616d820cfdaebp-58;
                p += 1*15;
            } else if (ax <= b2) {
                ch = 0x1.ea75575af6f09p+1;
                cl = -0x1.60155a9d1b256p-53;
                p += 2*15;
            } else {
                ch = 0x1.55365bc032467p+2;
                cl = 0x1.5c646a75d7539p-53;
                p += 3*15;
            }
        } else {
            if (ax <= b4) {
                ch = 0x1.c0ff5f3b47250p+2;
                cl = -0x1.b226d9d243827p-54;
                p += 4*15;
            } else if (ax <= b5) {
                ch = 0x1.112980f0b88a1p+3;
                cl = -0x1.63e17ec20a31dp-53;
                p += 5*15;
            } else if (ax <= b6) {
                ch = 0x1.458d0d0bdfc29p+3;
                cl = 0x1.02610a51562b6p-51;
                p += 6*15;
            } else {
                ch = 0x1.76979797ee5acp+3;
                cl = 0x1.9a84d3a5fedc2p-51;
                p += 7*15;
            }
        }

        ax = ax - ch - cl;

        ret = MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
              MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
              MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
              MATH_MAD(ax, MATH_MAD(ax,
              p[14], p[13]), p[12]),
              p[11]), p[10]), p[9]), p[8]),
              p[7]), p[6]), p[5]), p[4]),
              p[3]), p[2]), p[1]), p[0]);
    } else {
        // j1(x) ~ sqrt(2 / (pi*x)) * (P1(x) cos(x-3*pi/4) - Q1(x) sin(x-3*pi/4))
        double c;
        double s = MATH_PRIVATE(sincosb)(ax, 1, &c);
        const double sqrt2bypi = 0x1.9884533d43651p-1;
        if (ax > 0x1.0p+129)
            ret = MATH_DIV(sqrt2bypi * c, MATH_SQRT(ax));
        else
            ret = MATH_DIV(sqrt2bypi * (MATH_PRIVATE(pone)(ax)*c - MATH_PRIVATE(qone)(ax)*s), MATH_SQRT(ax));
        ret = BUILTIN_CLASS_F64(ax, CLASS_PINF) ? 0.0 : ret;
    }

    if (x < 0.0)
        ret = -ret;

    return ret;
}

