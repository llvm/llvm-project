/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

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
    double ret = MATH_MANGLE(atan)(vbyu);

    double t = piby2 - ret;
    ret = ax < ay ? t : ret;

    bool xneg = BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_NNOR|CLASS_NSUB|CLASS_NZER);

    t = pi - ret;
    ret = xneg ? t : ret;

    ret = BUILTIN_COPYSIGN_F64(ret, y);

    t = BUILTIN_COPYSIGN_F64(piby2, y);
    ret = BUILTIN_CLASS_F64(x, CLASS_NZER|CLASS_PZER) ? t : ret;

    t = BUILTIN_COPYSIGN_F64(pi, y);
    t = xneg ? t : y;
    ret = BUILTIN_CLASS_F64(y, CLASS_NZER|CLASS_PZER) ? t : ret;

    if (!FINITE_ONLY_OPT()) {
        t = xneg ? threepiby4 : piby4;
        t = BUILTIN_COPYSIGN_F64(t, y);
        ret = BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_PINF) &
              BUILTIN_CLASS_F64(y, CLASS_NINF|CLASS_PINF) ?
              t : ret;

        ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN) |
              BUILTIN_CLASS_F64(y, CLASS_SNAN|CLASS_QNAN) ?
              AS_DOUBLE(QNANBITPATT_DP64) : ret;
    }

    return ret;
}


