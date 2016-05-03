
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(trunc)(float x)
{
    return BUILTIN_TRUNC_F32(x);
}
