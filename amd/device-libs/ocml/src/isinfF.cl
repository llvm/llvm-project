
#include "mathF.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(isinf)(float x)
{
    return BUILTIN_CLASS_F32(x, CLASS_PINF|CLASS_NINF);
}
