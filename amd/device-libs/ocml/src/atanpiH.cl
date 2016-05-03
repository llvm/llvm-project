
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(atanpi)(half x)
{
    return MATH_MANGLE(atan)(x) * 0x1.45f306p-2h;
}

