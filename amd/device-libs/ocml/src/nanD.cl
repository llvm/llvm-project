
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(nan)(ulong nancode)
{
    return as_double((nancode & MANTBITS_DP64) | QNANBITPATT_DP64);
}

