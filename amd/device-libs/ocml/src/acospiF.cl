
#include "mathF.h"

CONSTATTR float
MATH_MANGLE(acospi)(float x)
{
    const float piinv = 0x1.45f306p-2f;

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
    float ztp = (2.0f*piinv) * MATH_MAD(s, u, s);
    float ztn = 1.0f - ztp;
    float z =  x < 0.0f ? ztn : ztp;
    float zb = MATH_MAD(-piinv, MATH_MAD(x, u, x), 0.5f);
    z = ax <= 0.5f ? zb : z;

    z = ax < 0x1.0p-28f ? 0.5f : z;
    z = ax > 1.0f ? AS_FLOAT(QNANBITPATT_SP32) : z;
    z = x == 1.0f ? 0.0f : z;
    z = x == -1.0f ? 1.0f : z;
    return z;
}

