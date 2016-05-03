
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(j0)(half x)
{
    return (half)MATH_UPMANGLE(j0)((float)x);
}

