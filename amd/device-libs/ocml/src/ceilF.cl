
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(ceil)(float x)
{
    return BUILTIN_CEIL_F32(x);
}
