
#include "mathH.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(signbit)(half x)
{
    return AS_SHORT(x) < 0;
}
