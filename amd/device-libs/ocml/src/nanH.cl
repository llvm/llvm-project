
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(nan)(ushort nancode)
{
    ushort h = (ushort)QNANBITPATT_HP16 | (nancode & (ushort)0x01ff);
    return AS_HALF(h);
}

