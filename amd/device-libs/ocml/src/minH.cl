/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half2
MATH_MANGLE2(min)(half2 x, half2 y)
{
    return BUILTIN_CMIN_2F16(x, y);
}

CONSTATTR INLINEATTR half
MATH_MANGLE(min)(half x, half y)
{
    return BUILTIN_CMIN_F16(x, y);
}

