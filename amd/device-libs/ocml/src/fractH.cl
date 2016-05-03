
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(fract)(half x, __private half *ip)
{
    *ip = BUILTIN_FLOOR_F16(x);
    return  BUILTIN_FRACTION_F16(x);
}

