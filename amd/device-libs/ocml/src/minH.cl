
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(min)(half x, half y)
{
    if (AMD_OPT()) {
        return BUILTIN_CMIN_F16(x, y);
    } else {
        return BUILTIN_MIN_F16(x, y);
    }
}

