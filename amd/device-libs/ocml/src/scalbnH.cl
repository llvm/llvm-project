
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(scalbn)(half x, int n)
{
    return MATH_MANGLE(ldexp)(x, n);
}

