/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float2
MATH_MANGLE(cacos)(float2 z)
{
    float2 a = MATH_MANGLE(cacosh)(z);
    bool b = AS_INT(z.y) < 0;
    return (float2)(b ? -a.y : a.y, b ? a.x : -a.x);
}

