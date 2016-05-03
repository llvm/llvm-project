
#include "mathD.h"

CONSTATTR double
MATH_MANGLE(atanpi)(double x)
{
    const double piinv = 0x1.45f306dc9c883p-2;
    return MATH_MANGLE(atan)(x) * piinv;
}

