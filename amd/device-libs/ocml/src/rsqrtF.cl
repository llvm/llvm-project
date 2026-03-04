/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(rsqrt)(float x)
{
    if (DAZ_OPT()) {
        return BUILTIN_AMDGPU_RSQRT_F32(x);
    } else {
        bool need_scale = x < 0x1p-126f;
        float scaled_input = need_scale ? 0x1.0p+24f * x : x;
        float result = BUILTIN_AMDGPU_RSQRT_F32(scaled_input);
        return need_scale ? result * 0x1.0p+12f : result;
    }
}

