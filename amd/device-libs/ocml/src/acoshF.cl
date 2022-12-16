/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float MATH_PRIVATE(lnep)(float2 a, int ea);

CONSTATTR float
MATH_MANGLE(acosh)(float x)
{
    bool b = x >= 0x1.0p+64f;
    float s = b ? 0x1.0p-64f : 1.0f;
    float sx = x * s;
    float2 a = add(sx, root2(sub(sqr(sx), s*s)));
    float z = MATH_PRIVATE(lnep)(a, b ? 64 : 0);

    if (!FINITE_ONLY_OPT()) {
        z = x == PINF_F32 ? x : z;
        z = x < 1.0f ? QNAN_F32 : z;
    }

    return z;
}

