
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(mad)(half a, half b, half c)
{
    return MATH_MAD(a, b, c);
}

