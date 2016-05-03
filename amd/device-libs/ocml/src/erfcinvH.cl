
#include "mathH.h"

INLINEATTR PUREATTR half
MATH_MANGLE(erfcinv)(half x)
{
    return (half)MATH_UPMANGLE(erfcinv)((float)x);
}

