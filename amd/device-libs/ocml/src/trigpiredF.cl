/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

INLINEATTR int
MATH_PRIVATE(trigpired)(float x, __private float *r)
{
    float t = 2.0f * BUILTIN_FRACTION_F32(0.5f * x);
    x = x > 1.0f ? t : x;
    t = BUILTIN_RINT_F32(2.0f * x);
    *r = MATH_MAD(t, -0.5f, x);
    return (int)t & 0x3;
}

