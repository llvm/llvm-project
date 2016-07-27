/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

INLINEATTR half
MATH_MANGLE(sinpi)(half x)
{
    half z = BUILTIN_COPYSIGN_F16(0.0h, x);
    x = BUILTIN_ABS_F16(x);
    half r = BUILTIN_FRACTION_F16(x);
    half txh = BUILTIN_TRUNC_F16(x) * 0.5h;
    half sgn = (txh != BUILTIN_TRUNC_F16(txh)) ^ BUILTIN_CLASS_F16(z, CLASS_NZER) ? -0.0h : 0.0h;

    half ret;

    // 2^52 <= |x| < Inf, the result is always integer
    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_PINF) ? AS_HALF((short)QNANBITPATT_HP16) : z;
    } else {
	ret = z;
    }

    // 0x1.0p-7 <= |x| < 2^10, result depends on which 0.25 interval
    // r < 1.0
    half a = 1.0h - r;
    int e = 0;

    //  r <= 0.75
    bool c = r <= 0.75h;
    half t = r - 0.5h;
    a = c ? t : a;
    e = c ? 1 : e;

    // r < 0.5
    c = r < 0.5h;
    t = 0.5h - r;
    a = c ? t : a;

    // r <= 0.25
    c = r <= 0.25h;
    a = c ? r : a;
    e = c ? 0 : e;

    const half pi = 0x1.921fb54442d18p+1h;
    half ca;
    half sa = MATH_PRIVATE(sincosred)(a * pi, &ca);

    half tret = BUILTIN_COPYSIGN_F16(e ? ca : sa, sgn);
    ret = x < 0x1.0p+10h ? tret : ret;

    return ret;
}

