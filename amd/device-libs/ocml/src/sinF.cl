/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

float
MATH_MANGLE(sin)(float x)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;

    struct redret r =  MATH_PRIVATE(trigred)(AS_FLOAT(ax));

#if defined EXTRA_PRECISION
    struct scret sc = MATH_PRIVATE(sincosred2)(r.hi, r.lo);
#else
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
#endif

    float s = (r.i & 1) != 0 ? sc.c : sc.s;
    s = AS_FLOAT(AS_INT(s) ^ (r.i > 1 ? 0x80000000 : 0) ^ (ix ^ ax));

    if (!FINITE_ONLY_OPT()) {
        s = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : s;
    }

    return s;
}

