/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(erfc)

CONSTATTR half
MATH_MANGLE(erfc)(half x)
{
    return (half)MATH_UPMANGLE(erfc)((float)x);
}

