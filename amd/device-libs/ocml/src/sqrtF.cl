/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

// 1ulp sqrt that handles denormals, should be used without
// -cl-fp32-correctly-rounded-divide-sqrt
static float sqrt_scale_denormal(float x) {
    bool need_scale = x < 0x1p-126f;
    float scaled = BUILTIN_FLDEXP_F32(x, need_scale ? 32 : 0);
    float sqrt_scaled = BUILTIN_AMDGPU_SQRT_F32(scaled);
    return BUILTIN_FLDEXP_F32(sqrt_scaled, need_scale ? -16 : 0);
}

CONSTATTR float
MATH_MANGLE(sqrt)(float x)
{
    if (CORRECTLY_ROUNDED_SQRT32()) {
        return MATH_SQRT(x);
    } else {
        if (DAZ_OPT())
            return BUILTIN_AMDGPU_SQRT_F32(x);
        return sqrt_scale_denormal(x);
    }
}

#define GEN(LN,UN) \
CONSTATTR float \
MATH_MANGLE(LN)(float x) \
{ \
    return BUILTIN_##UN##_F32(x); \
}

// GEN(sqrt_rte,SQRT_RTE)
// GEN(sqrt_rtn,SQRT_RTN)
// GEN(sqrt_rtp,SQRT_RTP)
// GEN(sqrt_rtz,SQRT_RTZ)

