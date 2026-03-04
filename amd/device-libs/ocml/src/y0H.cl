/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(y0)

CONSTATTR half
MATH_MANGLE(y0)(half x)
{
    return (half)MATH_UPMANGLE(y0)((float)x);
}

