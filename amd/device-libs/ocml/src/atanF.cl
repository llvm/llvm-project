/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

extern CONSTATTR float MATH_PRIVATE(atanred)(float);

CONSTATTR INLINEATTR float
MATH_MANGLE(atan)(float x)
{
    float v = BUILTIN_ABS_F32(x);
    bool g = v > 1.0f;

    float vi = MATH_FAST_RCP(v);
    v = g ? vi : v;

    float a = MATH_PRIVATE(atanred)(v);

    float y = MATH_MAD(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -a);
    a = g ? y : a;

    return BUILTIN_COPYSIGN_F32(a, x);
}

