
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

    const float pi = 0x1.921fb6p+1f;
    const float piby2 = 0x1.921fb6p+0f;

    float ax = BUILTIN_ABS_F32(x);

    float rt = MATH_MAD(-0.5f, ax, 0.5f);
    float ax2 = ax * ax;
    float r = ax > 0.5f ? rt : ax2;

    float u = r * MATH_MAD(r,
                           MATH_MAD(r,
                                    MATH_MAD(r,
                                             MATH_MAD(r,
                                                      MATH_MAD(r, 0x1.14e326p-5f, 0x1.17dda4p-6f),
                                                      0x1.fdcb1ep-6f),
                                             0x1.6d5902p-5f),
                                    0x1.33343cp-4f),
                           0x1.555554p-3f);

    float s = MATH_FAST_SQRT(r);
    float ztp = 2.0f * MATH_MAD(s, u, s);
    float ztn = pi - ztp;
    float z =  x < 0.0f ? ztn : ztp;
    float zb = piby2 - MATH_MAD(x, u, x);
    z = ax <= 0.5f ? zb : z;

    z = ax < 0x1.0p-28f ? piby2 : z;
    z = ax > 1.0f ? AS_FLOAT(QNANBITPATT_SP32) : z;
    z = x == 1.0f ? 0.0f : z;
    z = x == -1.0f ? pi : z;
    return z;
}

