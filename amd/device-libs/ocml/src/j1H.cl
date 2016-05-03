
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(j1)(half x)
{
    return (half)MATH_UPMANGLE(j1)((float)x);
}

