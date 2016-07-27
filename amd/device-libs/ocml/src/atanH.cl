/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(atan)(half x)
{
    const half piby2 = 0x1.921fb6p+0h;

    half v = BUILTIN_ABS_F16(x);

    // Reduce arguments 2^-7 <= |x| < 2^14
    half a = -1.0h;
    half b = v;
    half c = piby2; // atan(infinity)

    half ta = v - 1.5f;
    half tb = MATH_MAD(v, 1.5h, 1.0h);
    bool l = v <= 0x1.38p+1h; // 19/16 <= x < 39/16
    a = l ? ta : a;
    b = l ? tb : b;
    c = l ? 0x1.f730bep-1h : c; // atan(1.5)

    ta = v - 1.0h;
    tb = 1.0h + v;
    l = v <= 0x1.3p+0h; // 11/16 <= x < 19/16
    a = l ? ta : a;
    b = l ? tb : b;
    c = l ? 0x1.921fb6p-1h : c; // atan(1)

    ta = MATH_MAD(v, 2.0h, -1.0h);
    tb = 2.0h + v;
    l = v <= 0x1.6p-1h; // 7/16 <= x < 11/16
    a = l ? ta : a;
    b = l ? tb : b;
    c = l ? 0x1.dac670p-2f : c; // atan(0.5)

    l = v <= 0x1.cp-2h; // 2^-7 <= x < 7/16
    a = l ? v : a;
    b = l ? 1.0h : b;
    c = l ? 0.0h : c;

    half t = MATH_FAST_DIV(a, b);
    half t2 = t * t;

    half r = MATH_MAD(t*t2,
                MATH_MAD(t2,
                     MATH_MAD(t2,
                         MATH_MAD(t2, 0x1.c71c72p-4h, -0x1.24924ap-3h),
                         0x1.99999ap-3h),
                     -0x1.555556p-2h),
                t) + c;

    half ret;

    if (!FINITE_ONLY_OPT()) {
        ret = BUILTIN_CLASS_F16(x, CLASS_QNAN|CLASS_SNAN) ? x : piby2;
    } else {
        ret = piby2;
    }

    ret = v <= 0x1.0p+14h ? r : ret;
    ret = v < 0x1.0p-7h ? v : ret;

    return BUILTIN_COPYSIGN_F16(ret, x);
}

