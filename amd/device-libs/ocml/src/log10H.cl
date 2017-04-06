/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

PUREATTR UGEN(log10)

PUREATTR INLINEATTR half
MATH_MANGLE(log10)(half x)
{
    return (half)(BUILTIN_LOG2_F32((float)x) * 0x1.344136p-2f);
}

