/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double2
MATH_MANGLE(cacos)(double2 z)
{
    double2 a = MATH_MANGLE(cacosh)(z);
    bool b = AS_INT2(z.y).hi < 0;
    return (double2)(b ? -a.y : a.y, b ? a.x : -a.x);
}

