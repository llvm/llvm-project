
#include "mathH.h"

PUREATTR INLINEATTR half
MATH_MANGLE(exp10)(half x)
{
    if (AMD_OPT()) {
        return (half)BUILTIN_EXP2_F32((float)x * 0x1.a934f0p+1f);
    } else {
        return (half)MATH_UPMANGLE(exp10)((float)x);
    }
}

