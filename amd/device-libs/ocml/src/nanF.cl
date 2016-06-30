
#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(nan)(uint nancode)
{
    return AS_FLOAT(QNANBITPATT_SP32 | (nancode & 0xfffff));
}

