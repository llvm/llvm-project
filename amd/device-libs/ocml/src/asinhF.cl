
#include "mathF.h"

PUREATTR float
MATH_MANGLE(asinh)(float x)
{
    uint ux = AS_UINT(x);
    uint ax = ux & EXSIGNBIT_SP32;
    uint xsgn = ax ^ ux;
    float z;

    // |x| <= 2
    if (ax <= 0x40000000) {
        float t = x * x;
        float a = MATH_MAD(t,
                           MATH_MAD(t,
                                    MATH_MAD(t,
                                             MATH_MAD(t, -1.177198915954942694e-4f, -4.162727710583425360e-2f),
                                             -5.063201055468483248e-1f),
                                    -1.480204186473758321f),
                           -1.152965835871758072f);

        float b = MATH_MAD(t,
                           MATH_MAD(t,
                                    MATH_MAD(t,
                                             MATH_MAD(t, 6.284381367285534560e-2f, 1.260024978680227945f),
                                             6.582362487198468066f),
                                    11.99423176003939087f),
                           6.917795026025976739f);

        float q = MATH_FAST_DIV(a, b);
        z = MATH_MAD(x*t, q, x);
    } else {
        // Arguments greater than 1/sqrt(epsilon) in magnitude are
        // approximated by asinh(x) = ln(2) + ln(abs(x)), with sign of x
        // Arguments such that 4.0 <= abs(x) <= 1/sqrt(epsilon) are
        // approximated by asinhf(x) = ln(abs(x) + sqrt(x*x+1))
        // with the sign of x (see Abramowitz and Stegun 4.6.20)

        float absx = AS_FLOAT(ax);
        int hi = ax > 0x46000000U;
        float y = MATH_FAST_SQRT(MATH_MAD(absx, absx, 1.0f)) + absx;
        y = hi ? absx : y;
        float r = MATH_MANGLE(log)(y) + (hi ? 0x1.62e430p-1f : 0.0f);
        z = AS_FLOAT(xsgn | AS_UINT(r));
    }

    if (!FINITE_ONLY_OPT()) {
        z = ax < 0x39800000U | ax >= PINFBITPATT_SP32 ? x : z;
    } else {
        z = ax < 0x39800000U ? x : z;
    }

    return z;
}

