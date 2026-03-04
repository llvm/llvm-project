/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(erf)(float x)
{
    float ax = BUILTIN_ABS_F32(x);
    float ret;

    if (ax < 1.0f) {
        float t = ax*ax;
        float p = MATH_MAD(t, MATH_MAD(t, MATH_MAD(t, MATH_MAD(t,
                  MATH_MAD(t,
                      -0x1.268bc2p-11f, 0x1.420828p-8f), -0x1.b5937p-6f), 0x1.ce077cp-4f),
                      -0x1.81266p-2f), 0x1.06eba0p-3f);
        ret = BUILTIN_FMA_F32(ax, p, ax);
    } else {
        float p = MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax, MATH_MAD(ax,
                  MATH_MAD(ax, MATH_MAD(ax,
                      0x1.1d3156p-16f, -0x1.8d129p-12f), 0x1.f9a6d2p-9f), -0x1.8c3164p-6f),
                      0x1.b4e9c8p-4f), 0x1.4515fap-1f), 0x1.078e50p-3f);
        p = BUILTIN_FMA_F32(ax, p, ax);
        ret = 1.0f - MATH_MANGLE(exp)(-p);
    }

    ret = BUILTIN_COPYSIGN_F32(ret, x);
    return ret;
}

