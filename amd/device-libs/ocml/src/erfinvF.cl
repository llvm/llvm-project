/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

PUREATTR float
MATH_MANGLE(erfinv)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float w = -MATH_MANGLE(log)((1.0f-ax)*(1.0f+ax));

    float p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
            MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                0x1.e2cb10p-26f, 0x1.70966cp-22f), -0x1.d8e6aep-19f), -0x1.26b582p-18f), 0x1.ca65b6p-13f),
                                -0x1.48a810p-10f), -0x1.11c9dep-8f), 0x1.f91ec6p-3f), 0x1.805c5ep+0f);
    } else {
        w = MATH_SQRT(w) - 3.0f;
        p = MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
            MATH_MAD(w, MATH_MAD(w, MATH_MAD(w, MATH_MAD(w,
                -0x1.a3e136p-13f, 0x1.a76ad6p-14f), 0x1.61b8e4p-10f), -0x1.e17bcep-9f), 0x1.7824f6p-8f),
                                 -0x1.f38baep-8f), 0x1.354afcp-7f), 0x1.006db6p+0f), 0x1.6a9efcp+1f);
    }

    float ret = p*ax;

    if (!FINITE_ONLY_OPT()) {
        ret = ax > 1.0 ? AS_FLOAT(QNANBITPATT_SP32) : ret;
        ret = ax == 1.0 ? AS_FLOAT(PINFBITPATT_SP32) : ret;
    }

    return BUILTIN_COPYSIGN_F32(ret, x);
}

