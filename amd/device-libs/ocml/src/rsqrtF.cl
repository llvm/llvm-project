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
        float input_scale = need_scale ? 0x1.0p+24f : 1.0f;
        float output_scale = need_scale ? 0x1.0p+12f : 1.0f;
        return BUILTIN_AMDGPU_RSQRT_F32(x * input_scale) * output_scale;
    }
}

