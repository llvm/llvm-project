
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(lgamma)(half x)
{
    int s;
    return MATH_MANGLE(lgamma_r)(x, &s);
}

