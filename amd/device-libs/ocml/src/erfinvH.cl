/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(erfinv)

CONSTATTR half
MATH_MANGLE(erfinv)(half x)
{
    return (half)MATH_UPMANGLE(erfinv)((float)x);
}

