/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(tgamma)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 16.0) {
        double n, d;
        double y = x;
        if (x > 0.0) {
            n = 1.0;
            while (y > 2.5) {
                n = MATH_MAD(n, y, -n);
                y = y - 1.0;
                n = MATH_MAD(n, y, -n);
                y = y - 1.0;
            }
            if (y > 1.5) {
                n = MATH_MAD(n, y, -n);
                y = y - 1.0;
            }
            if (x >= 0.5)
                y = y - 1.0;
            d = x < 0.5 ? x : 1.0;
        } else {
            d = x;
            while (y < -1.5) {
                d = MATH_MAD(d, y, d);
                y = y + 1.0;
                d = MATH_MAD(d, y, d);
                y = y + 1.0;
            }
            if (y < -0.5) {
                d = MATH_MAD(d, y, d);
                y = y + 1.0;
            }
            n = 1.0;
        }
        double qt = MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y,
                    MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y,
                    MATH_MAD(y, MATH_MAD(y, MATH_MAD(y, MATH_MAD(y,
                    MATH_MAD(y,
                       -0x1.aed75feec7b9ap-23, 0x1.31854a0be3cd3p-20),
                       -0x1.5037d6a97a8b7p-20), -0x1.51d67f2cdbcfbp-16),
                       0x1.0c8ab2ac5112dp-13), -0x1.c364ce9b5e149p-13),
                       -0x1.317113a39f929p-10), 0x1.d919c501178a3p-8),
                       -0x1.3b4af282da690p-7), -0x1.59af103bf2cd0p-5),
                       0x1.5512320b432ccp-3), -0x1.5815e8fa28886p-5),
                       -0x1.4fcf4026afa24p-1), 0x1.2788cfc6fb61cp-1);

        ret = MATH_DIV(n, MATH_MAD(d, y*qt, d));
        ret = x == 0.0 ? BUILTIN_COPYSIGN_F64(PINF_F64, x) : ret;
        ret = x < 0.0 && BUILTIN_FRACTION_F64(x) == 0.0 ? QNAN_F64 : ret;
    } else {
        const double sqrt2pi = 0x1.40d931ff62706p+1;
        const double sqrtpiby2 = 0x1.40d931ff62706p+0;

        double t1 = MATH_MANGLE(powr)(ax, MATH_MAD(ax, 0.5, -0.25));
        double t2 = MATH_MANGLE(exp)(-ax);
        double xr = MATH_FAST_RCP(ax);
        double pt = MATH_MAD(xr, MATH_MAD(xr, MATH_MAD(xr, MATH_MAD(xr,
                    MATH_MAD(xr, MATH_MAD(xr,
                       -0x1.2b04c5ea74bbfp-11, 0x1.14869344f1d9bp-14),
                       0x1.9b3457156ffefp-11), -0x1.e1427e86ee097p-13),
                       -0x1.5f7266f67c4e0p-9), 0x1.c71c71c0f96adp-9),
                       0x1.5555555555a28p-4);

        if (x > 0.0) {
            double gt = sqrt2pi*t2*t1*t1;
            double g = MATH_MAD(gt, xr*pt, gt);
            ret = x > 0x1.573fae561f646p+7 ? PINF_F64 : g;
        } else {
            double s = -x * MATH_MANGLE(sinpi)(x);
            if (x > -170.5) {
                double d = s*t2*t1*t1;
                ret = MATH_DIV(sqrtpiby2, MATH_MAD(d, xr*pt, d));
            } else if (x > -184.0) {
                double d = t2*t1;
                ret = MATH_DIV(MATH_DIV(sqrtpiby2, MATH_MAD(d, xr*pt, d)), s*t1);
            } else
                ret = BUILTIN_COPYSIGN_F64(0.0, s);
            ret = BUILTIN_FRACTION_F64(x) == 0.0 || BUILTIN_ISNAN_F64(x) ? QNAN_F64 : ret;
        }
    }

    return ret;
}

