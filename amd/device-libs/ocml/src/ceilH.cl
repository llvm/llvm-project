/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half2
MATH_MANGLE2(ceil)(half2 x)
{
    return BUILTIN_CEIL_2F16(x);
}

CONSTATTR INLINEATTR half
MATH_MANGLE(ceil)(half x)
{
    return BUILTIN_CEIL_F16(x);
}

