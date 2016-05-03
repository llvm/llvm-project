
#include "mathH.h"

PUREATTR INLINEATTR half
MATH_MANGLE(log1p)(half x)
{
    half ret;
    if (AMD_OPT()) {
        ret = (half)(BUILTIN_LOG2_F32((float)x + 1.0f) * 0x1.62e430p-1f);
        half p = MATH_MAD(x, x*MATH_MAD(x, 0x1.555556p-2h, -0.5h), x);
        ret = BUILTIN_ABS_F16(x) < 0x1.0p-6h ? p : ret;
    } else {
        ret =  (half)MATH_UPMANGLE(log1p)((float)x);
    }

    return ret;
}

