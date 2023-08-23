/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

CONSTATTR struct redret
MATH_PRIVATE(trigred)(float x)
{
    if (x < SMALL_BOUND)
        return MATH_PRIVATE(trigredsmall)(x);
    else
        return MATH_PRIVATE(trigredlarge)(x);
}

