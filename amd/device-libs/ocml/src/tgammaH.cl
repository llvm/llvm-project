/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

UGEN(tgamma)

CONSTATTR half
MATH_MANGLE(tgamma)(half x)
{
    return (half)MATH_UPMANGLE(tgamma)((float)x);
}

