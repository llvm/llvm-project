/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half2
MATH_MANGLE2(fmin)(half2 x, half2 y)
{
    return BUILTIN_MIN_2F16(BUILTIN_CANONICALIZE_2F16(x), BUILTIN_CANONICALIZE_2F16(y));
}

CONSTATTR INLINEATTR half
MATH_MANGLE(fmin)(half x, half y)
{
    return BUILTIN_MIN_F16(BUILTIN_CANONICALIZE_F16(x), BUILTIN_CANONICALIZE_F16(y));
}

