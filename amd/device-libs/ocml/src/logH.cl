
#include "mathH.h"

PUREATTR INLINEATTR half
MATH_MANGLE(log)(half x)
{
    if (AMD_OPT()) {
        return (half)(BUILTIN_LOG2_F32((float)x) * 0x1.62e430p-1f);
    } else {
        return (half)MATH_UPMANGLE(log)((float)x);
    }
}

