
#include "mathF.h"

INLINEATTR float
MATH_MANGLE(lgamma)(float x)
{
    int s;
    return MATH_MANGLE(lgamma_r)(x, &s);
}

