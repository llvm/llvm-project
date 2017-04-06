/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(exp2)

PUREATTR INLINEATTR half
MATH_MANGLE(exp2)(half x)
{
    return BUILTIN_EXP2_F16(x);
}

