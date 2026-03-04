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

CONSTATTR float
MATH_MANGLE(cosh)(float x)
{
    x = BUILTIN_ABS_F32(x);
    float2 e = MATH_PRIVATE(epexpep)(sub(x, con(0x1.62e430p-1f, -0x1.05c610p-29f)));
    float2 c = fadd(e, ldx(rcp(e), -2));
    float z = c.hi;
    
    if (!FINITE_ONLY_OPT()) {
        z = x > 0x1.65a9f8p+6f ? PINF_F32 : z;
    }

    return z;
}

