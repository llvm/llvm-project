
#include "mathH.h"

PUREATTR INLINEATTR half
MATH_MANGLE(log10)(half x)
{
    if (AMD_OPT()) {
        return (half)(BUILTIN_LOG2_F32((float)x) * 0x1.344136p-2f);
    } else {
        return (half)MATH_UPMANGLE(log10)((float)x);
    }
}

