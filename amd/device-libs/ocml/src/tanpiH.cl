
#include "mathH.h"
#include "trigredH.h"

CONSTATTR INLINEATTR half
MATH_MANGLE(tanpi)(half x)
{
    return (half)MATH_UPMANGLE(tanpi)((float)x);
}

