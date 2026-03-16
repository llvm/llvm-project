/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

CONSTATTR float
MATH_MANGLE(tanpi)(float x)
{
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F32(x) ? QNAN_F32 : x;

    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F32(x));
    float t = MATH_PRIVATE(tanpired)(r.hi, r.i & 1);
    int flip = (((r.i == 1) | (r.i == 2)) & (r.hi == 0.0f)) ? SIGNBIT_SP32 : 0;
    t = AS_FLOAT((AS_INT(t) ^ flip) ^ (AS_INT(x) & SIGNBIT_SP32));

    return t;
}

