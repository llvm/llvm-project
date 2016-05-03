
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(trunc)(double x)
{
    return BUILTIN_TRUNC_F64(x);
}
