
#include "mathH.h"

INLINEATTR PUREATTR half
MATH_MANGLE(erfcx)(half x)
{
    return (half)MATH_UPMANGLE(erfcx)((float)x);
}

