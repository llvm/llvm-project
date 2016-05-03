
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(atanpi)(float x)
{
    const float pi = 0x1.921fb6p+1f;
    float ret = MATH_MANGLE(atan)(x);
    if (DAZ_OPT()) {
        ret = MATH_FAST_DIV(ret, pi);
    } else {
        ret = MATH_DIV(ret, pi);
    }
    return ret;
}
