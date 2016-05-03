
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(fabs)(double x)
{
    return BUILTIN_ABS_F64(x);
}

