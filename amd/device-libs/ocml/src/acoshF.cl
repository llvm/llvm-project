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
MATH_MANGLE(acosh)(float x)
{
    bool b = x >= 0x1.0p+64f;
    float s = b ? 0x1.0p-64f : 1.0f;
    float sx = x * s;
    float2 a = add(sx, root2(sub(sqr(sx), s*s)));
    float z = MATH_PRIVATE(lnep)(a) + (b ? 0x1.62e430p+5f : 0.0f);

    z = x == 1.0f ? 0.0f : z;

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_CLASS_F32(x, CLASS_PINF) ? x : z;
        z = x < 1.0f ? AS_FLOAT(QNANBITPATT_SP32) : z;
    }

    return z;
}

