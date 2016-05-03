
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

    const float piby2 = 0x1.921fb6p+0f;

    float ax = BUILTIN_ABS_F32(x);

    float tx = 0.5f * (1.0f - ax);
    float x2 = x*x;
    float r = ax >= 0.5f ? tx : x2;

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
    float ret = MATH_MAD(MATH_MAD(s, u, s), -2.0f, piby2);
    float xux = MATH_MAD(ax, u, ax);
    ret = ax < 0.5f ? xux : ret;

    ret = ax > 1.0f ? as_float(QNANBITPATT_SP32) : ret;
    ret = ax == 1.0f ? piby2 : ret;
    ret = BUILTIN_COPYSIGN_F32(ret, x);
    return ret;
}

