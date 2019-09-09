/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR half2
MATH_MANGLE2(fmax)(half2 x, half2 y)
{
    return BUILTIN_MAX_2F16(x, y);
}

CONSTATTR half
MATH_MANGLE(fmax)(half x, half y)
{
    return BUILTIN_MAX_F16(x, y);
}

