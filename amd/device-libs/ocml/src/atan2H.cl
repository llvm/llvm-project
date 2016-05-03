
#include "mathH.h"

CONSTATTR half
MATH_MANGLE(atan2)(half y, half x)
{
    const half pi = 0x1.921fb54442d18p+1h;
    const half piby2 = 0x1.921fb54442d18p+0h;
    const half piby4 = 0x1.921fb54442d18p-1h;
    const half threepiby4 = 0x1.2d97c7f3321d2p+1h;

    half ay = BUILTIN_ABS_F16(y);
    half ax = BUILTIN_ABS_F16(x);
    half u = BUILTIN_MAX_F16(ax, ay);
    half v = BUILTIN_MIN_F16(ax, ay);
    half vbyu = MATH_DIV(v, u);
    half ret = MATH_MANGLE(atan)(vbyu);

    half t = piby2 - ret;
    ret = ax < ay ? t : ret;

    bool xneg = BUILTIN_CLASS_F16(x, CLASS_NINF|CLASS_NNOR|CLASS_NSUB|CLASS_NZER);

    t = pi - ret;
    ret = xneg ? t : ret;

    ret = BUILTIN_COPYSIGN_F16(ret, y);

    t = BUILTIN_COPYSIGN_F16(piby2, y);
    ret = BUILTIN_CLASS_F16(x, CLASS_NZER|CLASS_PZER) ? t : ret;

    t = BUILTIN_COPYSIGN_F16(pi, y);
    t = xneg ? t : y;
    ret = BUILTIN_CLASS_F16(y, CLASS_NZER|CLASS_PZER) ? t : ret;

    if (!FINITE_ONLY_OPT()) {
        t = xneg ? threepiby4 : piby4;
        t = BUILTIN_COPYSIGN_F16(t, y);
        ret = BUILTIN_CLASS_F16(x, CLASS_NINF|CLASS_PINF) &
              BUILTIN_CLASS_F16(y, CLASS_NINF|CLASS_PINF) ?
              t : ret;

        ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN) |
              BUILTIN_CLASS_F16(y, CLASS_SNAN|CLASS_QNAN) ?
              as_half((short)QNANBITPATT_HP16) : ret;
    }

    return ret;
}

