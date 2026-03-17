/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(nextafter)(double x, double y)
{
    double up = MATH_MANGLE(succ)(x);
    double down = MATH_MANGLE(pred)(x);

    double ret = y;
    if (x < y)
        ret = up;
    if (x > y)
        ret = down;

    return BUILTIN_ISUNORDERED_F64(x, y) ? QNAN_F64 : ret;
}

