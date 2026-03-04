/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(acos)(float x)
{
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1 and arccos(-x) = arccos(x).
    // For denormal and small arguments arccos(x) = pi/2 to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    float ax = BUILTIN_ABS_F32(x);

    float rt = MATH_MAD(-0.5f, ax, 0.5f);
    float x2 = ax * ax;
    float r = ax > 0.5f ? rt : x2;

    float u = r * MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, MATH_MAD(r,
                  MATH_MAD(r,
                      0x1.38434ep-5f, 0x1.bf8bb4p-7f), 0x1.069878p-5f), 0x1.6c8362p-5f),
                      0x1.33379p-4f), 0x1.555558p-3f);

    float s = MATH_FAST_SQRT(r);
    float ztp = 2.0f * MATH_MAD(s, u, s);
    float ztn = MATH_MAD(0x1.ddcb02p+0f, 0x1.aee9d6p+0f, -ztp);
    float zt =  x < 0.0f ? ztn : ztp;
    float z = MATH_MAD(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -MATH_MAD(x, u, x));
    z = ax > 0.5f ? zt : z;

    return z;
}

