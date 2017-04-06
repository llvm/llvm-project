/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(log)

PUREATTR INLINEATTR half
MATH_MANGLE(log)(half x)
{
    return (half)(BUILTIN_LOG2_F32((float)x) * 0x1.62e430p-1f);
}

