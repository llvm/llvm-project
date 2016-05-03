
#include "mathF.h"
#include "trigredF.h"

INLINEATTR float
MATH_PRIVATE(sincosred)(float x, __private float *cp)
{
#if !defined USE_4_COEFF
    const float c0 =  0x1.55554ap-5f;
    const float c1 = -0x1.6c0c2cp-10f;
    const float c2 =  0x1.99ebdap-16f;

    const float s0 = -0x1.555544p-3f;
    const float s1 =  0x1.11072ep-7f;
    const float s2 = -0x1.994430p-13f;

    float x2 = x*x;
    float c = MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, c2, c1), c0), -0.5f), 1.0f);
    float s = MATH_MAD(x, x2*MATH_MAD(x2, MATH_MAD(x2, s2, s1), s0), x);
#else
    const float c0 =  0x1.555556p-5f;
    const float c1 = -0x1.6c16b2p-10f;
    const float c2 =  0x1.a00e98p-16f;
    const float c3 = -0x1.23c5e0p-22f;

    const float s0 = -0x1.555556p-3f;
    const float s1 =  0x1.11110ep-7f;
    const float s2 = -0x1.a0139ep-13f;
    const float s3 =  0x1.6dbc3ap-19f;

    float x2 = x*x;
    float x3 = x * x2;
    float r = 0.5f * x2;
    float t = 1.0f - r;
    float u = 1.0f - t;
    float v = u - r;

    float c = t + MATH_MAD(x2*x2, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, c3, c2), c1), c0), v);
    float s = MATH_MAD(x3, MATH_MAD(x2, MATH_MAD(x2, MATH_MAD(x2, s3, s2), s1), s0), x);
#endif

    *cp = c;
    return s;
}

