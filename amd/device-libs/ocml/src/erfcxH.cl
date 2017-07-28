
#include "mathH.h"

PUREATTR UGEN(erfcx)

PUREATTR half
MATH_MANGLE(erfcx)(half x)
{
    return (half)MATH_UPMANGLE(erfcx)((float)x);
}

