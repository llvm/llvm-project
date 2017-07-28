/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

CONSTATTR struct redret
MATH_PRIVATE(trigred)(double x)
{
    if (x < 0x1.0p+21)
        return MATH_PRIVATE(trigredsmall)(x);
    else
        return MATH_PRIVATE(trigredlarge)(x);
}

