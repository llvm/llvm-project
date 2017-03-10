/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR INLINEATTR half2
MATH_MANGLE2(mad)(half2 a, half2 b, half2 c)
{
    return MATH_MAD2(a, b, c);
}

CONSTATTR INLINEATTR half
MATH_MANGLE(mad)(half a, half b, half c)
{
    return MATH_MAD(a, b, c);
}

