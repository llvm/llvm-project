/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(i1)

half
MATH_MANGLE(i1)(half x)
{
    return (half)MATH_UPMANGLE(i1)((float)x);
}

