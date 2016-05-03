
#include "mathH.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(isinf)(half x)
{
    return BUILTIN_CLASS_F16(x, CLASS_PINF|CLASS_NINF);
}
