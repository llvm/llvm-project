
#include "mathF.h"

PUREATTR INLINEATTR float
MATH_MANGLE(atanh)(float x)
{
    uint ux = AS_UINT(x);
    uint ax = ux & EXSIGNBIT_SP32;
    uint xs = ux ^ ax;

    float z, t;

    if ((ax >= 0x3f000000U) & (ax < 0x3f800000U)) {
        // 1/2 <= |x| < 1
        t = AS_FLOAT(ax);
        t = MATH_FAST_DIV(2.0f*t, 1.0f - t);
        t = 0.5f * MATH_MANGLE(log1p)(t);
        z = AS_FLOAT(xs | AS_UINT(t));
    } else {
        // |x| < 1/2
        t = x * x;
        float a = MATH_MAD(MATH_MAD(0.92834212715e-2f, t, -0.28120347286e0f), t, 0.39453629046e0f);
        float b = MATH_MAD(MATH_MAD(0.45281890445e0f, t, -0.15537744551e1f), t, 0.11836088638e1f);
        float p = MATH_FAST_DIV(a, b);
        z = MATH_MAD(x*t, p, x);
        z = ax < 0x39000000U ? x : z;
    }

    if (!FINITE_ONLY_OPT()) {
        // |x| == 1
        t = AS_FLOAT(xs | PINFBITPATT_SP32);
        z = ax == 0x3f800000U ? t : z;

        // |x| > 1 or NaN
        z = ax > 0x3f800000U ? AS_FLOAT(QNANBITPATT_SP32) : z;
    }

    return z;
}

