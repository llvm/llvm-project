
#include "mathH.h"

PUREATTR half
MATH_MANGLE(tanh)(half x)
{
    // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
    // to the following three formulae:
    // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
    // 2.  (1 - (2/(exp(2*x) + 1 )))
    // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
    // but computationally, some formulae are better on some ranges.

    half ax = BUILTIN_ABS_F16(x);
    half ret;

    if (ax <= 0.55h) {
        half x2 = x*x;
        ret = MATH_MAD(x2,
                  MATH_MAD(x2,
                      MATH_MAD(x2, -0.053968253968253968254h, 0.13333333333333333333h),
                      -0.33333333333333333333h),
                  1.0f) * ax;
    } else if (ax <= 0x1.208p+2h) {
        half t = MATH_MANGLE(exp)(2.0h * ax);
        ret = 1.0h - MATH_FAST_DIV(2.0h, t + 1.0h);
    } else {
        ret = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN) ? x : 1.0h;
    }

    return BUILTIN_COPYSIGN_F16(ret, x);
}

