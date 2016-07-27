/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(erfcinv)(double y)
{
    double ret;

    if (y > 0.0625) {
        ret = MATH_MANGLE(erfinv)(1.0 - y);
    } else {
        const double c = 0.8862269254527580136490837416705725913988; // sqrt(pi)/2
        double t = MATH_RCP(MATH_SQRT(-MATH_MANGLE(log)(y)));
        double x;
        // We really need much more accurate rational approximations to avoid
        // the need to correct and the loss of accuracy for subnormal input
        if (y > 0x1.0p-333) {
            double a = MATH_MAD(t,
                           MATH_MAD(t,
                               MATH_MAD(t,
                                   MATH_MAD(t, -0x1.50c6bda236181p-3, 0x1.5c704ba7304bap-1),
                                   -0x1.20c9f12c389f8p+0),
                               0x1.61c6bc080422fp-1),
                           0x1.61f9ea3ab3a42p+0);
            double b = MATH_MAD(t, MATH_MAD(t, 1.0, 0x1.629e4fbf5e0bfp+0), 0x1.3d7dab206c232p-3);
            x = MATH_DIV(MATH_DIV(0x1.3d89481d73257p-3, t) + a, b);
        } else {
            double a = MATH_MAD(t, MATH_MAD(t, -0x1.133282b500c73p-1, 0x1.f230ec2315ba1p-1), 0x1.74655aea6112ap-2);
            double b = MATH_MAD(t, MATH_MAD(t, 1.0, 0x1.746dc3ed59b6ap-2), 0x1.414636de0e412p-7);
            x = MATH_DIV(MATH_DIV(0x1.4146a0a22696dp-7, t) + a, b);
        }

        if (y >= 0x1.0p-1022) {
            // We can't correct if input is subnormal
            double f = MATH_MANGLE(erfc)(x) - y;
            double dfi = -c * MATH_MANGLE(exp)(x*x);
            x -= dfi * f;

            f = MATH_MANGLE(erfc)(x) - y;
            dfi = -c * MATH_MANGLE(exp)(x*x);
            x -= dfi * f;
        }

        ret = x;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = (y < 0.0) | (y > 2.0) ? AS_DOUBLE(QNANBITPATT_DP64) : ret;
        ret = y == 0.0 ? AS_DOUBLE(PINFBITPATT_DP64) : ret;
        ret = y == 2.0 ? AS_DOUBLE(NINFBITPATT_DP64) : ret;
    }

    return ret;
}

