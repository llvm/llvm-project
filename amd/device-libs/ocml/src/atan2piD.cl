
#include "mathD.h"

CONSTATTR double
MATH_MANGLE(atan2pi)(double y, double x)
{
    const double piinv = 0x1.45f306dc9c883p-2;

    double ay = BUILTIN_ABS_F64(y);
    double ax = BUILTIN_ABS_F64(x);
    double u = BUILTIN_MAX_F64(ax, ay);
    double v = BUILTIN_MIN_F64(ax, ay);
    double vbyu = MATH_DIV(v, u);
    double ret = piinv * MATH_MANGLE(atan)(vbyu);

    double t = 0.5 - ret;
    ret = ax < ay ? t : ret;

    bool xneg = BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_NNOR|CLASS_NSUB|CLASS_NZER);

    t = 1.0 - ret;
    ret = xneg ? t : ret;

    ret = BUILTIN_COPYSIGN_F64(ret, y);

    t = BUILTIN_COPYSIGN_F64(0.5, y);
    ret = BUILTIN_CLASS_F64(x, CLASS_NZER|CLASS_PZER) ? t : ret;

    t = BUILTIN_COPYSIGN_F64(1.0, y);
    t = xneg ? t : y;
    ret = BUILTIN_CLASS_F64(y, CLASS_NZER|CLASS_PZER) ? t : ret;

    t = xneg ? 0.75 : 0.25;
    t = BUILTIN_COPYSIGN_F64(t, y);
    ret = BUILTIN_CLASS_F64(x, CLASS_NINF|CLASS_PINF) &
          BUILTIN_CLASS_F64(y, CLASS_NINF|CLASS_PINF) ?
          t : ret;

    ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN) |
          BUILTIN_CLASS_F64(y, CLASS_SNAN|CLASS_QNAN) ?
          as_double(QNANBITPATT_DP64) : ret;

    return ret;
}

