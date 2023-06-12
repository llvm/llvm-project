/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(log10)

CONSTATTR half
MATH_MANGLE(log10)(half x)
{
    return (half)(BUILTIN_AMDGPU_LOG2_F32((float)x) * 0x1.344136p-2f);
}

