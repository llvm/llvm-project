/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

extern CONSTATTR double MATH_PRIVATE(atanred)(double);

CONSTATTR double
MATH_MANGLE(atan2)(double y, double x)
{
    const double pi = 0x1.921fb54442d18p+1;
    const double piby2 = 0x1.921fb54442d18p+0;
    const double piby4 = 0x1.921fb54442d18p-1;
    const double threepiby4 = 0x1.2d97c7f3321d2p+1;

    double ay = BUILTIN_ABS_F64(y);
    double ax = BUILTIN_ABS_F64(x);
    double u = BUILTIN_MAX_F64(ax, ay);
    double v = BUILTIN_MIN_F64(ax, ay);
    double vbyu = MATH_DIV(v, u);

    double a = MATH_PRIVATE(atanred)(vbyu);

    bool xneg = AS_INT2(x).y < 0;

    double t = piby2 - a;
    a = ax < ay ? t : a;
    t = pi - a;
    a = xneg ? t : a;

    t = xneg ? pi : 0.0;
    a = y == 0.0 ? t : a;

    if (!FINITE_ONLY_OPT()) {
        t = xneg ? threepiby4 : piby4;
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

