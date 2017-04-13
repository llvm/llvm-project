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
MATH_MANGLE(log1p)(float x)
{
    float z = MATH_PRIVATE(lnep)(add(1.0, x));

    if (!FINITE_ONLY_OPT()) {
        z = BUILTIN_CLASS_F32(x, CLASS_PINF) ? x : z;
        z = x < -1.0f ? AS_FLOAT(QNANBITPATT_SP32) : z;
        z = x == -1.0f ? AS_FLOAT(NINFBITPATT_SP32) : z;
    }

    return BUILTIN_ABS_F32(x) < 0x1.0p-24f ? x : z;
}

