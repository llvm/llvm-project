/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(asin)

CONSTATTR half
MATH_MANGLE(asin)(half x)
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

    half ax = BUILTIN_ABS_F16(x);
    half r;

    if (ax <= 0.5h) {
        half s = x * x;
        half p = s * MATH_MAD(s, 0x1.828p-4h, 0x1.52p-3h);
        r = MATH_MAD(ax, p, ax);
    } else {
        float s = BUILTIN_MAD_F32((float)ax, -0.5f, 0.5f);
        float t = BUILTIN_SQRT_F32(s);
        float p = BUILTIN_MAD_F32(t, BUILTIN_MAD_F32(s, -0x1.82675ap-2f, -0x1.ff9f6p+0f), 0x1.921fb6p+0f);
        r = (half)p;
    }

    return BUILTIN_COPYSIGN_F16(r, x);
}

