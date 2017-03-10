/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(j0)

INLINEATTR half
MATH_MANGLE(j0)(half x)
{
    return (half)MATH_UPMANGLE(j0)((float)x);
}

