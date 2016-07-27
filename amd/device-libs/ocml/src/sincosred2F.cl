/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

INLINEATTR float
MATH_PRIVATE(sincosred2)(float x, float y, __private float *cp)
{
    const float c0 =  0x1.555556p-5f;
    const float c1 = -0x1.6c16b2p-10f;
    const float c2 =  0x1.a00e98p-16f;
    const float c3 = -0x1.23c5e0p-22f;

    const float s0 = -0x1.555556p-3f;
    const float s1 =  0x1.11110ep-7f;
    const float s2 = -0x1.a0139ep-13f;
    const float s3 =  0x1.6dbc3ap-19f;

    float x2 = x*x;
    float x3 = x * x2;
    float r = 0.5f * x2;
    float t = 1.0f - r;
    float u = 1.0f - t;
    float v = u - r;

    float cxy = t + MATH_MAD(x2*x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, c3, c2), c1), c0), MATH_MAD(x, -y, v));

    float sxy = MATH_MAD(x2, MATH_MAD(x2, s3, s2), s1);
    sxy = x - MATH_MAD(-x3, s0, MATH_MAD(x2, MATH_MAD(-x3, sxy, 0.5f*y), -y));

    *cp = cxy;
    return sxy;
}

