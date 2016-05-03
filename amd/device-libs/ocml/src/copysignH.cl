
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(copysign)(half x, half y)
{
    return BUILTIN_COPYSIGN_F16(x, y);
}

