/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern CONSTATTR float MATH_PRIVATE(atanred)(float);

CONSTATTR float
MATH_MANGLE(atan2)(float y, float x)
{
    const float pi = 0x1.921fb6p+1f;
    const float piby2 = 0x1.921fb6p+0f;
    const float piby4 = 0x1.921fb6p-1f;
    const float threepiby4 = 0x1.2d97c8p+1f;

    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    float v = BUILTIN_MIN_F32(ax, ay);
    float u = BUILTIN_MAX_F32(ax, ay);

    float vbyu;
    if (DAZ_OPT()) {
        float s = u > 0x1.0p+96f ? 0x1.0p-32f : 1.0f;
        vbyu = s * MATH_FAST_DIV(v, s*u);
    } else {
        vbyu = MATH_DIV(v, u);
    }

    float a = MATH_PRIVATE(atanred)(vbyu);

    float t = piby2 - a;
    a = ay > ax ? t : a;
    t = pi - a;
    a = x < 0.0f ? t : a;

    t = AS_INT(x) < 0 ? pi : 0.0f;
    a = y == 0.0f ? t : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        t = x < 0.0f ? threepiby4 : piby4;
        a = BUILTIN_ISINF_F32(x) & BUILTIN_ISINF_F32(y) ? t : a;

        // x or y is NaN
        a = BUILTIN_ISNAN_F32(x) | BUILTIN_ISNAN_F32(y) ?
            AS_FLOAT(QNANBITPATT_SP32) : a;
    }

    return BUILTIN_COPYSIGN_F32(a, y);
}

