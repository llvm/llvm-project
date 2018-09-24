
#include "mathH.h"

CONSTATTR UGEN(erfcx)

CONSTATTR half
MATH_MANGLE(erfcx)(half x)
{
    return (half)MATH_UPMANGLE(erfcx)((float)x);
}

