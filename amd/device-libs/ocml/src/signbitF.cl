
#include "mathF.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(signbit)(float x)
{
    return AS_INT(x) < 0;
}
