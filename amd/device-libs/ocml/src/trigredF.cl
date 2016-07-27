/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

INLINEATTR int
#if defined EXTRA_PRECISION
MATH_PRIVATE(trigred)(__private float *r, __private float *rr, float x)
#else
MATH_PRIVATE(trigred)(__private float *r, float x)
#endif
{
    if (x < SMALL_BOUND)
#if defined EXTRA_PRECISION
        return MATH_PRIVATE(trigredsmall)(r, rr, x);
#else
        return MATH_PRIVATE(trigredsmall)(r, x);
#endif
    else
#if defined EXTRA_PRECISION
        return MATH_PRIVATE(trigredlarge)(r, rr, x);
#else
        return MATH_PRIVATE(trigredlarge)(r, x);
#endif
}

