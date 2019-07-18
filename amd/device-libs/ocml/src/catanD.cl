/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double2
MATH_MANGLE(catan)(double2 z)
{
    double2 a = MATH_MANGLE(catanh)((double2)(-z.y, z.x));
    return (double2)(a.y, -a.x);
}

