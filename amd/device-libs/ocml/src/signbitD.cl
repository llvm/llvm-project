
#include "mathD.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(signbit)(double x)
{
    return as_int2(x).hi < 0;
}

