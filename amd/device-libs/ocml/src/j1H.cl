/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(j1)

half
MATH_MANGLE(j1)(half x)
{
    return (half)MATH_UPMANGLE(j1)((float)x);
}

