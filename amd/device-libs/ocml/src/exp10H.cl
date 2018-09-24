/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR UGEN(exp10)

CONSTATTR half
MATH_MANGLE(exp10)(half x)
{
    return (half)BUILTIN_EXP2_F32((float)x * 0x1.a934f0p+1f);
}

