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
MATH_MANGLE(log1p)(float x)
{
    float z = MATH_PRIVATE(lnep)(add(1.0f, x), 0);

    if (!FINITE_ONLY_OPT()) {
        z = x == PINF_F32 ? x : z;
        z = x < -1.0f ? QNAN_F32 : z;
        z = x == -1.0f ? NINF_F32 : z;
    }

    return BUILTIN_ABS_F32(x) < 0x1.0p-24f ? x : z;
}

