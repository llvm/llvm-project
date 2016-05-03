
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(fabs)(half x)
{
    return BUILTIN_ABS_F16(x);
}

