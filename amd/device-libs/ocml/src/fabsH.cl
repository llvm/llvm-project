/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(fabs)(half2 x)
{
    return BUILTIN_ABS_2F16(x);
}

CONSTATTR half
MATH_MANGLE(fabs)(half x)
{
    return BUILTIN_ABS_F16(x);
}

