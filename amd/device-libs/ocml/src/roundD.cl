
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(round)(double x)
{
    double t = BUILTIN_TRUNC_F64(x);
    double d = BUILTIN_ABS_F64(x - t);
    double o = BUILTIN_COPYSIGN_F64(1.0, x);
    return t + (d >= 0.5 ? o : 0.0);
}

