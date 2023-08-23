/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(asin)(float x)
{
    // Computes arcsin(x).
    // The argument is first reduced by noting that arcsin(x)
    // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
    // For denormal and small arguments arcsin(x) = x to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arcsin(x) = x + x^3*R(x^2)
    // where R(x^2) is a polynomial minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
    // together with the above polynomial approximation, and
    // reconstruct the terms carefully.

    float ax = BUILTIN_ABS_F32(x);
    float tx = MATH_MAD(ax, -0.5f, 0.5f);
    float x2 = x*x;
    float r = ax >= 0.5f ? tx : x2;

    float u = r * MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, MATH_MAD(r,
                  MATH_MAD(r,
                      0x1.38434ep-5f, 0x1.bf8bb4p-7f), 0x1.069878p-5f), 0x1.6c8362p-5f),
                      0x1.33379p-4f), 0x1.555558p-3f);

    float s = MATH_FAST_SQRT(r);
    float ret = MATH_MAD(0x1.ddcb02p-1f, 0x1.aee9d6p+0f, -2.0f*MATH_MAD(s, u, s));

    float xux = MATH_MAD(ax, u, ax);
    ret = ax < 0.5f ? xux : ret;

    return BUILTIN_COPYSIGN_F32(ret, x);
}

