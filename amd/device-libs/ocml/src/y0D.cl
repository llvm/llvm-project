/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern double MATH_PRIVATE(sinb)(double, int, double);
extern CONSTATTR double MATH_PRIVATE(bp0)(double);
extern CONSTATTR double MATH_PRIVATE(ba0)(double);

CONSTATTR double
MATH_MANGLE(y0)(double x)
{
    const double b0  = 0.3125;
    const double b1  = 0.4375;
    const double b2  = 0.5625;
    const double b3  = 0.6875;
    const double b4  = 0.8125;
    const double b5  = 1.0;
    const double b6  = 1.25;
    const double b7  = 1.625;
    const double b8  = 2.0;
    const double b9  = 2.53125;
    const double b10 = 3.0;
    const double b11 = 3.484375;
    const double b12 = 4.703125;
    const double b13 = 6.265625;
    const double b14 = 7.84375;
    const double b15 = 9.421875;
    const double b16 = 10.984375;
    const double b17 = 12.546875;
    double ret;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(double, p, M64_Y0);
        double ch, cl;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0;
                    cl = 0.0;
                } else if (x < b1) {
                    ch = 0x1.4p-2;
                    cl = 0.0;
                    p += 1*15;
                } else if (x < b2) {
                    ch = 0x1.cp-2;
                    cl = 0.0;
                    p += 2*15;
                } else if (x < b3) {
                    ch = 0x1.2p-1;
                    cl = 0.0;
                    p += 3*15;
                } else {
                    ch = 0x1.6p-1;
                    cl = 0.0;
                    p += 4*15;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.c982eb8d417eap-1;
                    cl = 0x1.ea9d270347f83p-56;
                    p += 5*15;
                } else if (x < b6) {
                    ch = 0x1.p+0;
                    cl = 0.0;
                    p += 6*15;
                } else if (x < b7) {
                    ch = 0x1.4p+0;
                    cl = 0.0;
                    p += 7*15;
                } else {
                    ch = 0x1.ap+0;
                    cl = 0.0;
                    p += 8*15;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.193bed4dff243p+1;
                    cl = -0x1.bd1e50d219bfdp-55;
                    p += 9*15;
                } else if (x < b10) {
                    ch = 0x1.44p+1;
                    cl = 0.0;
                    p += 10*15;
                } else if (x < b11) {
                    ch = 0x1.8p+1;
                    cl = 0.0;
                    p += 11*15;
                } else if (x < b12) {
                    ch = 0x1.fa9534d98569cp+1;
                    cl = -0x1.f06ae7804384ep-54;
                    p += 12*15;
                } else {
                    ch = 0x1.5b7fe4e87b02ep+2;
                    cl = 0x1.dfe7bac228e8cp-52;
                    p += 13*15;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.c581dc4e72103p+2;
                    cl = -0x1.9774a495f56cfp-54;
                    p += 14*15;
                } else if (x < b15) {
                    ch = 0x1.13127ae6169b4p+3;
                    cl = 0x1.479cc068d9046p-52;
                    p += 15*15;
                } else if (x < b16) {
                    ch = 0x1.471d735a47d58p+3;
                    cl = -0x1.cb49ff791c495p-51;
                    p += 16*15;
                } else {
                    ch = 0x1.77f9138d43206p+3;
                    cl = 0x1.0fc786ce0608p-55;
                    p += 17*15;
                }
            }
        }

        ret = 0.0;
        if (x < b0) {
            ret = 0x1.45f306dc9c883p-1 * MATH_MANGLE(j0)(x) * MATH_MANGLE(log)(x);
            x = x*x;
        }

        x = x - ch - cl;
        ret += MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
               MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
               MATH_MAD(x, MATH_MAD(x, MATH_MAD(x, MATH_MAD(x,
               MATH_MAD(x, MATH_MAD(x,
               p[14], p[13]), p[12]),
               p[11]), p[10]), p[9]), p[8]),
               p[7]), p[6]), p[5]), p[4]),
               p[3]), p[2]), p[1]), p[0]);

    } else {
        double r = MATH_RCP(x);
        double r2 = r*r;
        double p = MATH_PRIVATE(bp0)(r2) * r;
        ret = 0x1.9884533d43651p-1 * MATH_FAST_SQRT(r) * MATH_PRIVATE(ba0)(r2) * MATH_PRIVATE(sinb)(x, 0, p);
        ret = BUILTIN_CLASS_F64(x, CLASS_PINF) ? 0.0 : ret;
    }

    return ret;
}

