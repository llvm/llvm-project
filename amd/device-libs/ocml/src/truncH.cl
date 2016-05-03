
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(trunc)(half x)
{
    return BUILTIN_TRUNC_F16(x);
}
