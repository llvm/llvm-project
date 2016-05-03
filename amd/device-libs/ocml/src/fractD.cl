
#include "mathD.h"

INLINEATTR double
MATH_MANGLE(fract)(double x, __private double *ip)
{
    *ip = BUILTIN_FLOOR_F64(x);
    return BUILTIN_FRACTION_F64(x);
}

