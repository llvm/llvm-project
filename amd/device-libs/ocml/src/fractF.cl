/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

float
MATH_MANGLE(fract)(float x, __private float *ip)
{
    float i = BUILTIN_FLOOR_F32(x);

    float f;
    if (__oclc_ISA_version < 800) {
        f = BUILTIN_MIN_F32(x - i, 0x1.fffffep-1f);
        if (!FINITE_ONLY_OPT()) {
            f = BUILTIN_ISNAN_F32(x) ? x : f;
            f = BUILTIN_ISINF_F32(x) ? 0.0f : f;
        }
    } else {
        f = BUILTIN_FRACTION_F32(x);
    }

    *ip = i;
    return f;
}

