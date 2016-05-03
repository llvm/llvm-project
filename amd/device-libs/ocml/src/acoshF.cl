
#include "mathF.h"

PUREATTR float
MATH_MANGLE(acosh)(float x)
{
    uint ux = as_uint(x);

    // Arguments greater than 1/sqrt(epsilon) in magnitude are
    // approximated by acosh(x) = ln(2) + ln(x)
    // For 2.0 <= x <= 1/sqrt(epsilon) the approximation is
    // acosh(x) = ln(x + sqrt(x*x-1)) */
    int high = ux > 0x46000000U;
    int med = ux > 0x40000000U;

    float w = x - 1.0f;
    float s = MATH_MAD(w, w, 2.0f*w);
    float t = MATH_MAD(x, x, -1.0f);
    float r = MATH_FAST_SQRT(med ? t : s) + (med ? x : w);
    float v = (high ? x : r) - (med ? 1.0f : 0.0f);
    float z = MATH_MANGLE(log1p)(v) + (high ? 0x1.62e430p-1f : 0.0f);

    if (!FINITE_ONLY_OPT()) {
        z = ux >= PINFBITPATT_SP32 ? x : z;
        z = x < 1.0f ? as_float(QNANBITPATT_SP32) : z;
    }

    return z;
}

