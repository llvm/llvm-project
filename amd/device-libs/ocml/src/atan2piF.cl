/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern CONSTATTR float MATH_PRIVATE(atanpired)(float);

CONSTATTR float
MATH_MANGLE(atan2pi)(float y, float x)
{
    const float pi = 0x1.921fb6p+1f;

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

    float a = MATH_PRIVATE(atanpired)(vbyu);

    float at = 0.5f - a;
    a = ay > ax ? at : a;
    at = 1.0f - a;
    a = x < 0.0f ? at : a;

    at = AS_INT(x) < 0 ? 1.0f : 0.0f;
    a = y == 0.0f ? at : a;

    if (!FINITE_ONLY_OPT()) {
        // x and y are +- Inf
        at = x < 0.0f ? 0.75f : 0.25f;
        a = (BUILTIN_ISINF_F32(x) & BUILTIN_ISINF_F32(y)) ? at : a;

        // x or y is NaN
        a = BUILTIN_ISUNORDERED_F32(x, y) ? QNAN_F32 : a;
    }

    return BUILTIN_COPYSIGN_F32(a, y);
}
