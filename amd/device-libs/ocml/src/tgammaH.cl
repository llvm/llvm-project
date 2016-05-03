
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(tgamma)(half x)
{
    return (half)MATH_UPMANGLE(tgamma)((float)x);
}

