
#include "mathH.h"

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

    const half piinv = 0x1.45f306p-2h;

    half ax = BUILTIN_ABS_F16(x);

    half rt = MATH_MAD(-0.5h, ax, 0.5h);
    half ax2 = ax * ax;
    half r = ax > 0.5h ? rt : ax2;

    half u = r * MATH_MAD(r, MATH_MAD(r, 0x1.5e98c8p-5h, 0x1.0ba2eep-4h), 0x1.561ee8p-3h);

    half s = MATH_FAST_SQRT(r);
    half ztp = (2.0h*piinv) * MATH_MAD(s, u, s);
    half ztn = 1.0h - ztp;
    half z =  x < 0.0h ? ztn : ztp;
    half zb = MATH_MAD(-piinv, MATH_MAD(x, u, x), 0.5h);
    z = ax <= 0.5h ? zb : z;

    z = ax < 0x1.0p-11h ? 0.5h : z;
    z = x == 1.0h ? 0.0h : z;
    z = x == -1.0h ? 1.0h : z;
    z = ax > 1.0h ? as_half((short)QNANBITPATT_HP16) : z;
    return z;
}

