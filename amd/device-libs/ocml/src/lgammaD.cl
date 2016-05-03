
#include "mathD.h"

INLINEATTR double
MATH_MANGLE(lgamma)(double x)
{
    int s;
    return MATH_MANGLE(lgamma_r)(x, &s);
}

