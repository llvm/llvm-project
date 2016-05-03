
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(ceil)(double x)
{
    return BUILTIN_CEIL_F64(x);
}
