/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(exp2)

CONSTATTR half
MATH_MANGLE(exp2)(half x)
{
    return BUILTIN_EXP2_F16(x);
}

