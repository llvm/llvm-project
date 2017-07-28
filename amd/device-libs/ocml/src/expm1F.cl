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
MATH_MANGLE(expm1)(float x)
{
    float2 e = sub(MATH_PRIVATE(epexpep)(con(x, 0.0f)), 1.0f);
    float z = e.hi;
    
    if (!FINITE_ONLY_OPT()) {
        z = x > 0x1.62e42ep+6f ? AS_FLOAT(PINFBITPATT_SP32) : z;
    }

    z = x < -17.0f ? -1.0f : z;

    return z;
}

