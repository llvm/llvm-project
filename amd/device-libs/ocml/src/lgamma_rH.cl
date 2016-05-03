
#include "mathH.h"

INLINEATTR half
MATH_MANGLE(lgamma_r)(half x, __private int *signp)
{
    return (half)MATH_UPMANGLE(lgamma_r)((float)x, signp);
}

