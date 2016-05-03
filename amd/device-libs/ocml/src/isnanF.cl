
#include "mathF.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(isnan)(float x)
{
    return BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN);
}
