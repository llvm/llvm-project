
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(max)(double x, double y)
{
    if (AMD_OPT()) {
        return BUILTIN_CMAX_F64(x, y);
    } else {
        return BUILTIN_MAX_F64(x, y);
    }
}

