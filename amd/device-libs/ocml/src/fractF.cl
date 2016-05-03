
#include "mathF.h"

INLINEATTR float
MATH_MANGLE(fract)(float x, __private float *ip)
{
    *ip = BUILTIN_FLOOR_F32(x);
    return  BUILTIN_FRACTION_F32(x);
}

