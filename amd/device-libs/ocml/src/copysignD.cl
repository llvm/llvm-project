
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(copysign)(double x, double y)
{
    return BUILTIN_COPYSIGN_F64(x, y);
}

