
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(nan)(uint nancode)
{
    return as_float(QNANBITPATT_SP32 | (nancode & 0xfffff));
}

