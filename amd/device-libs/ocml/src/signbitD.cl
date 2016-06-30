
#include "mathD.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(signbit)(double x)
{
    return AS_INT2(x).hi < 0;
}

