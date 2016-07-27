/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

INLINEATTR int
MATH_PRIVATE(trigred)(__private double *r, __private double *rr, double x)
{
    if (x < 0x1.0p+21)
        return MATH_PRIVATE(trigredsmall)(r, rr, x);
    else
        return MATH_PRIVATE(trigredlarge)(r, rr, x);
}

