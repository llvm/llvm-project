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

CONSTATTR float
MATH_MANGLE(atanh)(float x)
{
    float y = BUILTIN_ABS_F32(x);
    float2 a = fdiv(fadd(1.0f, y), fsub(1.0f, y));
    float z = 0.5f * MATH_PRIVATE(lnep)(a);
    z = y < 0x1.0p-12f ? y : z;

    if (!FINITE_ONLY_OPT()) {
        z = y > 1.0f ? AS_FLOAT(QNANBITPATT_SP32) : z;
        z = y == 1.0f ? AS_FLOAT(PINFBITPATT_SP32) : z;
    }

    return BUILTIN_COPYSIGN_F32(z, x);
}

