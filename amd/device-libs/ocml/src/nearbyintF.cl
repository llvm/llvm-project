
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(nearbyint)(float x)
{
    return BUILTIN_RINT_F32(x);
}

