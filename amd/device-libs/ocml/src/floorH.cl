
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(floor)(half x)
{
    return BUILTIN_FLOOR_F16(x);
}

