/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE(ccos)(float2 z)
{
    return MATH_MANGLE(ccosh)((float2)(-z.y, z.x));
}

