
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(scalbn)(float x, int n)
{
    return MATH_MANGLE(ldexp)(x, n);
}

