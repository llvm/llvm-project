
#include "mathH.h"

CONSTATTR INLINEATTR int
MATH_MANGLE(isnormal)(half x)
{
    return BUILTIN_CLASS_F16(x, CLASS_PNOR|CLASS_NNOR);
}
