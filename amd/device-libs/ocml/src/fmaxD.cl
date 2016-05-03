
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(fmax)(double x, double y)
{
    return BUILTIN_MAX_F64(BUILTIN_CANONICALIZE_F64(x), BUILTIN_CANONICALIZE_F64(y));
}

