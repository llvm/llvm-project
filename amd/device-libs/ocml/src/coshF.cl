/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
extern PUREATTR float MATH_PRIVATE(sinhcosh)(float y, int which);

PUREATTR float
MATH_MANGLE(cosh)(float x)
{
    float y = BUILTIN_ABS_F32(x);

    float z;
    if (y < 0x1.0a2b24p+3f) {
        z = MATH_PRIVATE(sinhcosh)(y, 1);
    } else {
        float t = MATH_MANGLE(exp)(y - 0x1.62e500p-1f);
        z = MATH_MAD(0x1.a0210ep-18f, t, t);
    }

    if (!FINITE_ONLY_OPT()) {
        z = y >= 0x1.65a9fap+6f ? AS_FLOAT(PINFBITPATT_SP32) : z;
        z = BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN) ? x : z;
    }

    return z;
}

