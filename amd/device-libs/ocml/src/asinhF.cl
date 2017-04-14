/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float MATH_PRIVATE(lnep)(float2 x);

CONSTATTR INLINEATTR float
MATH_MANGLE(asinh)(float x)
{
    float y = BUILTIN_ABS_F32(x);
    bool b = y >= 0x1.0p+64f;
    float s = b ? 0x1.0p-64f : 1.0f;
    float sy = y * s;
    float2 a = add(sy, root2(add(sqr(sy), s*s)));
    float z = MATH_PRIVATE(lnep)(a) + (b ? 0x1.62e430p+5f : 0.0);
    z = y < 0x1.0p-12f ? y : z;

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_CLASS_F32(y, CLASS_PINF) ? y : z;
    }

    return BUILTIN_COPYSIGN_F32(z, x);
}

