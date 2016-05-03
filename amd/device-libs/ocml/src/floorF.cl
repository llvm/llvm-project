
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(floor)(float x)
{
    return BUILTIN_FLOOR_F32(x);
}

