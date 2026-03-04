/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(minmag)(float x, float y)
{
    float ret = BUILTIN_MIN_F32(x, y);
    float ax = BUILTIN_ABS_F32(x);
    float ay = BUILTIN_ABS_F32(y);
    ret = ax < ay ? x : ret;
    ret = ay < ax ? y : ret;
    return ret;
}

