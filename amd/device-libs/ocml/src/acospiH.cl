/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(acospi)

CONSTATTR half
MATH_MANGLE(acospi)(half x)
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

    const half piinv = 0x1.46p-2h;

    half ax = BUILTIN_ABS_F16(x);

    half rt = MATH_MAD(-0.5h, ax, 0.5h);
    half x2 = ax * ax;
    half r = ax > 0.5h ? rt : x2;

    half u = r * MATH_MAD(r, 0x1.0b8p-5h, 0x1.a7cp-5h);

    half s = MATH_FAST_SQRT(r);
    half ztp = 2.0h * MATH_MAD(s, u, piinv*s);
    half ztn = 1.0h - ztp;
    half zt =  x < 0.0h ? ztn : ztp;
    half z = 0.5h - MATH_MAD(x, u, piinv*x);
    z = ax > 0.5h ? zt : z;

    return z;
}

