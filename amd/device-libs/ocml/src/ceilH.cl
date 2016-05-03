
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(ceil)(half x)
{
    return BUILTIN_CEIL_F16(x);
}
