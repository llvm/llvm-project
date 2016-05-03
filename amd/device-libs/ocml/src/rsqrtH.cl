
#include "mathH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(rsqrt)(half x)
{
    if (AMD_OPT()) {
        return BUILTIN_RSQRT_F16(x);
    } else {
        return MATH_FAST_RCP(MATH_FAST_SQRT(x));
    }
}

