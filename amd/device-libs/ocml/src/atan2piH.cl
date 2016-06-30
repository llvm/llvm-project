
#include "mathH.h"

CONSTATTR half
MATH_MANGLE(atan2pi)(half y, half x)
{
    const half piinv = 0x1.45f306dc9c883p-2h;

    half ay = BUILTIN_ABS_F16(y);
    half ax = BUILTIN_ABS_F16(x);
    half u = BUILTIN_MAX_F16(ax, ay);
    half v = BUILTIN_MIN_F16(ax, ay);
    half vbyu = MATH_DIV(v, u);
    half ret = piinv * MATH_MANGLE(atan)(vbyu);

    half t = 0.5h - ret;
    ret = ax < ay ? t : ret;

    bool xneg = BUILTIN_CLASS_F16(x, CLASS_NINF|CLASS_NNOR|CLASS_NSUB|CLASS_NZER);

    t = 1.0h - ret;
    ret = xneg ? t : ret;

    ret = BUILTIN_COPYSIGN_F16(ret, y);

    t = BUILTIN_COPYSIGN_F16(0.5h, y);
    ret = BUILTIN_CLASS_F16(x, CLASS_NZER|CLASS_PZER) ? t : ret;

    t = BUILTIN_COPYSIGN_F16(1.0h, y);
    t = xneg ? t : y;
    ret = BUILTIN_CLASS_F16(y, CLASS_NZER|CLASS_PZER) ? t : ret;

    if (!FINITE_ONLY_OPT()) {
        t = xneg ? 0.75h : 0.25h;
        t = BUILTIN_COPYSIGN_F16(t, y);
        ret = BUILTIN_CLASS_F16(x, CLASS_NINF|CLASS_PINF) &
              BUILTIN_CLASS_F16(y, CLASS_NINF|CLASS_PINF) ?
              t : ret;

        ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN) |
              BUILTIN_CLASS_F16(y, CLASS_SNAN|CLASS_QNAN) ?
              AS_HALF((short)QNANBITPATT_HP16) : ret;
    }

    return ret;
}

