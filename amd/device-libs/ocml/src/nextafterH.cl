/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"

CONSTATTR BGEN(nextafter)

CONSTATTR half
MATH_MANGLE(nextafter)(half x, half y)
{
    half up = MATH_MANGLE(succ)(x);
    half down = MATH_MANGLE(pred)(x);

    half ret = y;
    if (x < y)
        ret = up;
    if (x > y)
        ret = down;

    return BUILTIN_ISUNORDERED_F16(x, y) ? QNAN_F16 : ret;
}

