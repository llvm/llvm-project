
#include "mathH.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(signbit)(half x)
{
    return as_short(x) < 0;
}
