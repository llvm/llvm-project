
#include "mathH.h"
#include "trigredH.h"

INLINEATTR half
MATH_MANGLE(cospi)(half x)
{
    x = BUILTIN_ABS_F16(x);
    half r = BUILTIN_FRACTION_F16(x);
    half txh = BUILTIN_TRUNC_F16(x) * 0.5h;
    half sgn = txh != BUILTIN_TRUNC_F16(txh) ? -1.0h : 1.0h;
    half ret;

    // 2^11 <= |x| < Inf, the result is always even integer
    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_PINF) ? AS_HALF((short)QNANBITPATT_HP16) : 1.0h;
    } else {
	ret = 1.0h;
    }

    // 2^10 <= |x| < 2^11, the result is always integer
    ret = x < 0x1.0p+51 ? sgn : ret;

    // 0x1.0p-7 <= |x| < 2^11, result depends on which 0.25 interval

    // r < 1.0
    half a = 1.0h - r;
    int e = 1;
    half s = -sgn;

    // r <= 0.75
    bool c = r <= 0.75h;
    half t = r - 0.5h;
    a = c ? t : a;
    e = c ? 0 : e;

    // r < 0.5
    c = r < 0.5h;
    t = 0.5h - r;
    a = c ? t : a;
    s = c ? sgn : s;

    // r <= 0.25
    c = r <= 0.25h;
    a = c ? r : a;
    e = c ? 1 : e;

    const half pi = 0x1.921fb54442d18p+1h;
    half ca;
    half sa = MATH_PRIVATE(sincosred)(a * pi, &ca);

    half tret = BUILTIN_COPYSIGN_F16(e ? ca : sa, s);
    ret = x < 0x1.0p+10h ? tret : ret;

    return ret;
}

