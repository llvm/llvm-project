/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 x);

CONSTATTR INLINEATTR float
MATH_MANGLE(sinh)(float x)
{
    float y = BUILTIN_ABS_F32(x);
    float2 e = MATH_PRIVATE(epexpep)(sub(y, con(0x1.62e430p-1f, -0x1.05c610p-29f)));
    float2 s = fsub(e, ldx(rcp(e), -2));
    float z = s.hi;

    if (!FINITE_ONLY_OPT()) {
        z = y > 0x1.65a9f8p+6f ? AS_FLOAT(PINFBITPATT_SP32) : z;
    }

    z = y < 0x1.0p-12f ? y : z;
    return BUILTIN_COPYSIGN_F32(z, x);
}

