
#include "mathH.h"

INLINEATTR PUREATTR half
MATH_MANGLE(erf)(half x)
{
    return (half)MATH_UPMANGLE(erf)((float)x);
}

