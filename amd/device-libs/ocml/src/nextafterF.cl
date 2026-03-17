/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(nextafter)(float x, float y)
{
    float up = MATH_MANGLE(succ)(x);
    float down = MATH_MANGLE(pred)(x);

    float ret = y;
    if (x < y)
        ret = up;
    if (x > y)
        ret = down;

    return BUILTIN_ISUNORDERED_F32(x, y) ? QNAN_F32 : ret;
}
