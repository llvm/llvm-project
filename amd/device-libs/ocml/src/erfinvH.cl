
#include "mathH.h"

INLINEATTR PUREATTR half
MATH_MANGLE(erfinv)(half x)
{
    return (half)MATH_UPMANGLE(erfinv)((float)x);
}

