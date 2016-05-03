
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(y1)(half x)
{
    return (half)MATH_UPMANGLE(y1)((float)x);
}

