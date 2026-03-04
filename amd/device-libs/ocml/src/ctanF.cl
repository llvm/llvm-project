/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE(ctan)(float2 z)
{
    float2 r = MATH_MANGLE(ctanh)((float2)(-z.y, z.x));
    return (float2)(r.y, -r.x);
}

