/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(asinpi)(float x)
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

    const float piinv = 0x1.45f306p-2f;

    float ax = BUILTIN_ABS_F32(x);

    float tx = MATH_MAD(ax, -0.5f, 0.5f);
    float x2 = ax * ax;
    float r = ax >= 0.5f ? tx : x2;

    float u = r * MATH_MAD(r, MATH_MAD(r, MATH_MAD(r, MATH_MAD(r,
                  MATH_MAD(r,
                      -0x1.3f1c6cp-8f, 0x1.2ac560p-6f), 0x1.80aab4p-8f), 0x1.e53378p-7f),
                      0x1.86680ap-6f), 0x1.b29c5ap-5f);

    float s = MATH_FAST_SQRT(r);
    float ret = MATH_MAD(-2.0f, MATH_MAD(s, u, piinv*s), 0.5f);
    float xux = MATH_MAD(piinv, ax, ax*u);
    ret = ax >= 0.5f ? ret : xux;

    return BUILTIN_COPYSIGN_F32(ret, x);
}

