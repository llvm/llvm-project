/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

PUREATTR double
MATH_MANGLE(acosh)(double x)
{
    double ret;

    if (x < 1.0) {
        ret = AS_DOUBLE(QNANBITPATT_DP64);
    } else if (x < 2.5) {
        // Use x+sqrt(x^2-1) = 1 + t+sqrt(t^2+2t), where t=x-1
        // Use extra precision
        // We can drop about .1 ulp by raising the bound to 4
        double t = x - 1.0;
        double u1 = t * 2.0;

        // (t,0) * (t,0) -> (v1, v2)
        double v1 = t * t;
        double v2 = BUILTIN_FMA_F64(t, t, -v1);

        // (u1,0) + (v1,v2) -> (w1,w2)
        double r = u1 + v1;
        double s = (((u1 - r) + v1) + v2);
        double w1 = r + s;
        double w2 = (r - w1) + s;

        // sqrt(w1,w2) -> (u1,u2)
        double p1 = MATH_SQRT(w1);
        double a1 = p1*p1;
        double a2 = BUILTIN_FMA_F64(p1, p1, -a1);
        double temp = (((w1 - a1) - a2) + w2);
        double p2 = MATH_FAST_DIV(temp * 0.5, p1);
        u1 = p1 + p2;
        double u2 = (p1 - u1) + p2;

        // (u1,u2) + (t,0) -> (r1,r2)
        r = u1 + t;
        s = ((u1 - r) + t) + u2;
        // r1 = r + s;
        // r2 = (r - r1) + s;
        // t = r1 + r2;
        t = r + s;

        ret = MATH_MANGLE(log1p)(t);
        ret = x == 1.0 ? 0.0 : ret;
    } else if (x < 0x1.0p+28) {
        // We could use x+sqrt(x^2-1) = 2x - 1/(x+sqrt(x^2-1)) instead
        ret = MATH_MANGLE(log)(x + MATH_FAST_SQRT(BUILTIN_FMA_F64(x, x, -1.0)));
    } else {
        const double ln2 = 0x1.62e42fefa39efp-1;
        ret = MATH_MANGLE(log)(x) + ln2;
    }

    return ret;
}

