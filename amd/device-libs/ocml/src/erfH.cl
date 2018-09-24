/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(erf)

CONSTATTR half
MATH_MANGLE(erf)(half x)
{
    return (half)MATH_UPMANGLE(erf)((float)x);
}

