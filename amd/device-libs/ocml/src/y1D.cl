/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern double MATH_PRIVATE(sinb)(double, int, double);
extern CONSTATTR double MATH_PRIVATE(bp1)(double);
extern CONSTATTR double MATH_PRIVATE(ba1)(double);

CONSTATTR double
MATH_MANGLE(y1)(double x)
{
    const double b0 = 0.5;
    const double b1 = 0.625;
    const double b2 = 0.75;
    const double b3 = 0.9375;
    const double b4 = 1.21875;
    const double b5 = 1.53125;
    const double b6 = 1.84375;
    const double b7 = 2.078125;
    const double b8 = 2.3125;
    const double b9 = 2.734375;
    const double b10 = 3.15625;
    const double b11 = 4.203125;
    const double b12 = 4.6875;
    const double b13 = 6.1875;
    const double b14 = 7.76953125;
    const double b15 = 9.359375;
    const double b16 = 10.9375;
    const double b17 = 12.5625;

    double ret;

    if (x <= b17) {
        // Ty to maintain relative accuracy here

        USE_TABLE(double, p, M64_Y1);
        double ch, cl;

        if (x < b8) {
            if (x < b4) {
                if (x < b0) {
                    ch = 0.0;
                    cl = 0.0;
                    p += 0*15;
                } else if (x < b1) {
                    ch = 0x1.0p-1;
                    cl = 0.0;
                    p += 1*15;
                } else if (x < b2) {
                    ch = 0x1.4p-1;
                    cl = 0.0;
                    p += 2*15;
                } else if (x < b3) {
                    ch = 0x1.8p-1;
                    cl = 0.0;
                    p += 3*15;
                } else {
                    ch = 0x1.ep-1;
                    cl = 0.0;
                    p += 4*15;
                }
            } else {
                if (x < b5) {
                    ch = 0x1.38p+0;
                    cl = 0.0;
                    p += 5*15;
                } else if (x < b6) {
                    ch = 0x1.88p+0;
                    cl = 0.0;
                    p += 6*15;
                } else if (x < b7) {
                    ch = 0x1.d8p+0;
                    cl = 0.0;
                    p += 7*15;
                } else {
                    ch = 0x1.193bed4dff243p+1;
                    cl = -0x1.bd1e50d219bfdp-55;
                    p += 8*15;
                }
            }
        } else {
            if (x < b13) {
                if (x < b9) {
                    ch = 0x1.28p+1;
                    cl = 0.0;
                    p += 9*15;
                } else if (x < b10) {
                    ch = 0x1.5ep+1;
                    cl = 0.0;
                    p += 10*15;
                } else if (x < b11) {
                    ch = 0x1.d76d4affba175p+1;
                    cl = 0x1.3bac0714e4129p-58;
                    p += 11*15;
                } else if (x < b12) {
                    ch = 0x1.0dp+2;
                    cl = 0.0;
                    p += 12*15;
                } else {
                    ch = 0x1.5b7fe4e87b02ep+2;
                    cl = 0x1.dfe7bac228e8cp-52;
                    p += 13*15;
                }
            } else {
                if (x < b14) {
                    ch = 0x1.bc41890588553p+2;
                    cl = 0x1.7960b6b1c46acp-53;
                    p += 14*15;
                } else if (x < b15) {
                    ch = 0x1.13127ae6169b4p+3;
                    cl = 0x1.479cc068d9046p-52;
                    p += 15*15;
                } else if (x < b16) {
                    ch = 0x1.43f2ee51e8c7ep+3;
                    cl = 0x1.8f4ba5d68e44p-51;
                    p += 16*15;
                } else {
                    ch = 0x1.77f9138d43206p+3;
                    cl = 0x1.0fc786ce0608p-55;
                    p += 17*15;
                }
            }
        }

        double x2 = x*x;
        double xs = x - ch - cl;
        double t = x < b0 ? x2 : xs;
        ret = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
              MATH_MAD(t, MATH_MAD(t,
              p[14], p[13]), p[12]),
              p[11]), p[10]), p[9]), p[8]),
              p[7]), p[6]), p[5]), p[4]),
              p[3]), p[2]), p[1]), p[0]);

        if (x < b0) {
            const double twobypi = 0x1.45f306dc9c883p-1;
            if (x < 0x1.0p-33)
                ret = MATH_DIV(-twobypi, BUILTIN_ABS_F64(x));
            else
                ret = MATH_MAD(ret, x, twobypi*(MATH_MANGLE(j1)(x) * MATH_MANGLE(log)(x) - MATH_RCP(x)));
            ret = x < 0.0 ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        }
    } else {
        double r = MATH_RCP(x);
        double r2 = r*r;
        double p = MATH_PRIVATE(bp1)(r2) * r;
        ret = 0x1.9884533d43651p-1 * MATH_FAST_SQRT(r) * MATH_PRIVATE(ba1)(r2) * MATH_PRIVATE(sinb)(x, 1, p);
        ret = BUILTIN_CLASS_F64(x, CLASS_PINF) ? 0.0 : ret;
    }

    return ret;
}

