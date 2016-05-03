
#include "mathF.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(signbit)(float x)
{
    return as_int(x) < 0;
}
