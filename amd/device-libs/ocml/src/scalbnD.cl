
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(scalbn)(double x, int n)
{
    return MATH_MANGLE(ldexp)(x, n);
}

