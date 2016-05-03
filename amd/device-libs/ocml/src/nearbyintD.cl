
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(nearbyint)(double x)
{
    return BUILTIN_RINT_F64(x);
}

