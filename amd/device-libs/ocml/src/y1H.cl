/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(y1)

half
MATH_MANGLE(y1)(half x)
{
    return (half)MATH_UPMANGLE(y1)((float)x);
}

