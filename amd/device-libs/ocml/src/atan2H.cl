
/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

extern CONSTATTR half MATH_PRIVATE(atanred)(half);

CONSTATTR BGEN(atan2)

CONSTATTR half
MATH_MANGLE(atan2)(half y, half x)
{
    const half pi = 0x1.921fb6p+1h;
    const half piby2 = 0x1.921fb6p+0h;
    const half piby4 = 0x1.921fb6p-1h;
    const half threepiby4 = 0x1.2d97c8p+1h;

    half ax = BUILTIN_ABS_F16(x);
    half ay = BUILTIN_ABS_F16(y);
    half v = BUILTIN_MIN_F16(ax, ay);
    half u = BUILTIN_MAX_F16(ax, ay);

    half vbyu = MATH_DIV(v, u);

    half a = MATH_PRIVATE(atanred)(vbyu);

    half t = piby2 - a;
    a = ay > ax ? t : a;
    t = pi - a;
    a = x < 0.0h ? t : a;

    t = AS_SHORT(x) < 0 ? pi : 0.0h;
    a = y == 0.0h ? t : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        t = x < 0.0h ? threepiby4 : piby4;
        a = BUILTIN_ISINF_F16(x) & BUILTIN_ISINF_F16(y) ? t : a;

        // x or y is NaN
        a = BUILTIN_ISNAN_F16(x) | BUILTIN_ISNAN_F16(y) ?
            AS_HALF((short)QNANBITPATT_HP16) : a;
    }

    return BUILTIN_COPYSIGN_F16(a, y);
}

