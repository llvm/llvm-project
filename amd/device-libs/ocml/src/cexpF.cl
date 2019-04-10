/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE(cexp)(float2 z)
{
    float x = z.s0;
    float y = z.s1;
    float cy;
    float sy = MATH_MANGLE(sincos)(y, &cy);
    bool g = x > 88.0f;
    float ex = MATH_MANGLE(exp)(x - (g ? 1.0f : 0.0f));
    const float e1 =  0x1.5bf0a8p+1f;
    cy *= g ? e1 : 1.0f;
    sy *= g ? e1 : 1.0f;
    float rr = ex * cy;
    float ri = ex * sy;

    if (!FINITE_ONLY_OPT()) {
        bool b = BUILTIN_CLASS_F32(y, CLASS_NINF|CLASS_PINF|CLASS_QNAN|CLASS_SNAN);
        if (BUILTIN_CLASS_F32(x, CLASS_NINF)) {
            rr = 0.0f;
            ri = b ? 0.0f : ri;
        }
        if (BUILTIN_CLASS_F32(x, CLASS_PINF)) {
            rr = b ? AS_FLOAT(PINFBITPATT_SP32) : rr;
            ri = b ? AS_FLOAT(QNANBITPATT_SP32) : ri;
            ri = y == 0.0f ? y : ri;
        }
        ri = (BUILTIN_ISNAN_F32(x) & (y == 0.0f)) ? y : ri;
    }

    return (float2)(rr, ri);
}

