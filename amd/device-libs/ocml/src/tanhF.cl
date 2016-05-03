
#include "mathF.h"

PUREATTR float
MATH_MANGLE(tanh)(float x)
{
    // The definition of tanh(x) is sinh(x)/cosh(x), which is also equivalent
    // to the following three formulae:
    // 1.  (exp(x) - exp(-x))/(exp(x) + exp(-x))
    // 2.  (1 - (2/(exp(2*x) + 1 )))
    // 3.  (exp(2*x) - 1)/(exp(2*x) + 1)
    // but computationally, some formulae are better on some ranges.

    float ax = MATH_MANGLE(fabs)(x);
    float ret;

    if (ax <= 0.55f) {
        float x2 = x*x;

        ret = MATH_MAD(x2,
                  MATH_MAD(x2,
                      MATH_MAD(x2,
                          MATH_MAD(x2,
                              MATH_MAD(x2,
                                  MATH_MAD(x2,
                                      MATH_MAD(x2, -0.00145583438705131826825f, 0.00359212803657248101693f),
                                      -0.00886323552990219656886f),
                                  0.021869488536155202822f),
                              -0.053968253968253968254f),
                          0.13333333333333333333f),
                      -0.33333333333333333333f),
                  1.0f) * ax;
    } else if (ax <= 9.3f) {
        float t = MATH_MANGLE(exp)(2.0f * ax);
        ret = 1.0f - MATH_FAST_DIV(2.0f, t + 1.0f);
    } else {
        ret = MATH_MANGLE(isnan)(x) ? x : 1.0f;
    }

    return MATH_MANGLE(copysign)(ret, x);
}

