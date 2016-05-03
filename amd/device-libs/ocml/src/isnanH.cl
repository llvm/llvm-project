
#include "mathH.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(isnan)(half x)
{
    return BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN);
}

