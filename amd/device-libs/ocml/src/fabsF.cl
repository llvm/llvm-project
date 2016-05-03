
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(fabs)(float x)
{
    return BUILTIN_ABS_F32(x);
}

