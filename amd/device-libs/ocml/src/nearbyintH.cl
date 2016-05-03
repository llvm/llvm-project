
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(nearbyint)(half x)
{
    return BUILTIN_RINT_F16(x);
}

