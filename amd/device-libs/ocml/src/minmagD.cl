/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(minmag)(double x, double y)
{
    double ret = BUILTIN_MIN_F64(x, y);
    double ax = BUILTIN_ABS_F64(x);
    double ay = BUILTIN_ABS_F64(y);
    ret = ax < ay ? x : ret;
    ret = ay < ax ? y : ret;
    return ret;
}

