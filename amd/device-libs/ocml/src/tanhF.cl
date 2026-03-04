/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

#if defined EXTRA_ACCURACY
#define FLOAT_SPECIALIZATION
#include "ep.h"

extern CONSTATTR float2 MATH_PRIVATE(epexpep)(float2 x);
#endif

CONSTATTR float
MATH_MANGLE(tanh)(float x)
{
    float y = BUILTIN_ABS_F32(x);

#if defined EXTRA_ACCURACY
    float2 e = MATH_PRIVATE(epexpep)(con(y, 0.0f));
    float2 ei = rcp(e);
    float2 t = fdiv(fsub(e, ei), fadd(e, ei));
    float z = t.hi;

    z = y > 9.0f ? 1.0f : z;
    z = y < 0x1.0p-13f ? y : z;
#else
    float z;
    if (y < 0.625f) {
        float y2 = y*y;
        float p = MATH_MAD(y2, MATH_MAD(y2, MATH_MAD(y2, MATH_MAD(y2,
                    -0x1.758e7ap-8f, 0x1.521192p-6f), -0x1.b8389cp-5f),
                    0x1.110704p-3f), -0x1.555532p-2f);
        z = MATH_MAD(y2, y*p, y);
    } else {
        float t = MATH_MANGLE(exp)(2.0f * y);
        z = MATH_MAD(-2.0f, MATH_FAST_RCP(t + 1.0f), 1.0f);
    }
#endif

    return BUILTIN_COPYSIGN_F32(z, x);
}

