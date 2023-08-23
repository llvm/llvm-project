/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double2
MATH_MANGLE(ctan)(double2 z)
{
    double2 r = MATH_MANGLE(ctanh)((double2)(-z.y, z.x));
    return (double2)(r.y, -r.x);
}

