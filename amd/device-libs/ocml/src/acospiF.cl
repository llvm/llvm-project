/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(acospi)(float x)
{
    const float piinv = 0x1.45f306p-2f;

    float ax = BUILTIN_ABS_F32(x);

    float rt = MATH_MAD(-0.5f, ax, 0.5f);
    float x2 = ax * ax;
    float r = ax > 0.5f ? rt : x2;

    float u = r * MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, 
                  MATH_MAD(r, 
                      -0x1.3f1c6cp-8f, 0x1.2ac560p-6f), 0x1.80aab4p-8f), 0x1.e53378p-7f),
                      0x1.86680ap-6f), 0x1.b29c5ap-5f);

    float s = MATH_FAST_SQRT(r);
    float ztp = 2.0f * MATH_MAD(s, u, piinv*s);
    float ztn = 1.0f - ztp;
    float zt =  x < 0.0f ? ztn : ztp;
    float z = 0.5f - MATH_MAD(x, u, piinv*x);
    z = ax > 0.5f ? zt : z;

    return z;
}

