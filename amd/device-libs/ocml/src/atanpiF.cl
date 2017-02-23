/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern CONSTATTR float MATH_PRIVATE(atanred)(float);

CONSTATTR INLINEATTR float
MATH_MANGLE(atanpi)(float x)
{
    const float pi = 0x1.921fb6p+1f;

    float v = BUILTIN_ABS_F32(x);
    bool g = v > 1.0f;

    float vi = MATH_FAST_RCP(v);
    v = g ? vi : v;

    float a = MATH_PRIVATE(atanred)(v);

    if (DAZ_OPT()) {
        a = MATH_FAST_DIV(a, pi);
    } else {
        a = MATH_DIV(a, pi);
    }

    float y = 0.5f - a;
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F32(a, x);
}


