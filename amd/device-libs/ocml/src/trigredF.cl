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
    // Prefer nans use the small path. The large path has elidable nan checks
    // implied by the condition and the small does not.
    if (x >= SMALL_BOUND)
        return MATH_PRIVATE(trigredlarge)(x);
    else
        return MATH_PRIVATE(trigredsmall)(x);
}

