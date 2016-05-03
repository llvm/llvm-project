
#include "mathH.h"

INLINEATTR PUREATTR half
MATH_MANGLE(erfc)(half x)
{
    return (half)MATH_UPMANGLE(erfc)((float)x);
}

