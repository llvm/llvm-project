/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern CONSTATTR float MATH_PRIVATE(atanpired)(float);

CONSTATTR INLINEATTR float
MATH_MANGLE(atanpi)(float x)
{
    float v = BUILTIN_ABS_F32(x);
    bool g = v > 1.0f;

    float vi = MATH_FAST_RCP(v);
    v = g ? vi : v;

    float a = MATH_PRIVATE(atanpired)(v);

    float y = 0.5f - a;
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F32(a, x);
}


