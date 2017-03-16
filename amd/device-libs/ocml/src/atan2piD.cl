/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double MATH_PRIVATE(atanpired)(double);

CONSTATTR double
MATH_MANGLE(atan2pi)(double y, double x)
{
    const double pi = 0x1.921fb54442d18p+1;

    double ay = BUILTIN_ABS_F64(y);
    double ax = BUILTIN_ABS_F64(x);
    double u = BUILTIN_MAX_F64(ax, ay);
    double v = BUILTIN_MIN_F64(ax, ay);
    double vbyu = MATH_DIV(v, u);

    double a = MATH_PRIVATE(atanpired)(vbyu);

    bool xneg = AS_INT2(x).y < 0;

    double t = 0.5 - a;
    a = ax < ay ? t : a;
    t = 1.0 - a;
    a = xneg ? t : a;

    t = xneg ? 1.0 : 0.0;
    a = y == 0.0 ? t : a;

    if (!FINITE_ONLY_OPT()) {
        t = xneg ? 0.75 : 0.25;
        t = BUILTIN_COPYSIGN_F64(t, y);
        a = BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_PINF) &
              BUILTIN_CLASS_F64(y, CLASS_NINF|CLASS_PINF) ?
              t : a;

        a = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN) |
              BUILTIN_CLASS_F64(y, CLASS_SNAN|CLASS_QNAN) ?
              AS_DOUBLE(QNANBITPATT_DP64) : a;
    }

    return BUILTIN_COPYSIGN_F64(a, y);
}

