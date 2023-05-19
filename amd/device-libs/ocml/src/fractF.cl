/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

float
MATH_MANGLE(fract)(float x, __private float *ip)
{
    *ip = BUILTIN_FLOOR_F32(x);
    return BUILTIN_FRACTION_F32(x);
}

