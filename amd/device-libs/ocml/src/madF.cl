/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE2(mad)(float2 a, float2 b, float2 c)
{
    return MATH_MAD2(a, b, c);
}

CONSTATTR float
MATH_MANGLE(mad)(float a, float b, float c)
{
    return MATH_MAD(a, b, c);
}

