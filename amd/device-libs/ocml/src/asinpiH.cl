/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(asinpi)

CONSTATTR half
MATH_MANGLE(asinpi)(half x)
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

    const half piinv = 0x1.45f306p-2h;

    half ax = BUILTIN_ABS_F16(x);

    half tx = 0.5h * (1.0h - ax);
    half x2 = x*x;
    half r = ax >= 0.5h ? tx : x2;

    half u = r * MATH_MAD(r, MATH_MAD(r, 0x1.6db6dcp-5h, 0x1.333334p-4h), 0x1.555556p-3h);

    half s = MATH_FAST_SQRT(r);
    half ret = MATH_MAD(-2.0h*piinv, MATH_MAD(s, u, s), 0.5h);
    half xux = piinv*MATH_MAD(ax, u, ax);
    ret = ax < 0.5h ? xux : ret;

    ret = ax > 1.0h ? AS_HALF((short)QNANBITPATT_HP16) : ret;
    ret = ax == 1.0h ? 0.5h : ret;
    ret = BUILTIN_COPYSIGN_F16(ret, x);
    return ret;
}

