
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(rint)(half x)
{
    return BUILTIN_RINT_F16(x);
}
