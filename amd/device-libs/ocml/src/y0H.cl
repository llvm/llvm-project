
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(y0)(half x)
{
    return (half)MATH_UPMANGLE(y0)((float)x);
}

