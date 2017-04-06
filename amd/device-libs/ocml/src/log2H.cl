/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(log2)

PUREATTR INLINEATTR half
MATH_MANGLE(log2)(half x)
{
    return BUILTIN_LOG2_F16(x);
}

