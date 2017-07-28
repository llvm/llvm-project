/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

float
MATH_MANGLE(tan)(float x)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;

    struct redret r = MATH_PRIVATE(trigred)(AS_FLOAT(ax));

#if defined EXTRA_PRECISION
    float t = MATH_PRIVATE(tanred)(r.hi + r.lo, r.i & 1);
#else
    float t = MATH_PRIVATE(tanred)(r.hi, r.i & 1);
#endif

    t = AS_FLOAT(AS_INT(t) ^ (ix ^ ax));

    if (!FINITE_ONLY_OPT()) {
        t = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : t;
    }

    return t;
}

