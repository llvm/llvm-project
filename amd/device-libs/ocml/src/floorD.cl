
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(floor)(double x)
{
    return BUILTIN_FLOOR_F64(x);
}

