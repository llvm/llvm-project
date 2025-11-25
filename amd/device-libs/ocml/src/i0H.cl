/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(i0)

half
MATH_MANGLE(i0)(half x)
{
    return (half)MATH_UPMANGLE(i0)((float)x);
}

