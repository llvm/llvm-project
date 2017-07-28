/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

float
MATH_MANGLE(cos)(float x)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;

    struct redret r = MATH_PRIVATE(trigred)(AS_FLOAT(ax));

#if defined EXTRA_PRECISION
    struct scret sc = MATH_PRIVATE(sincosred2)(r.hi, r.lo);
#else
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
#endif
    sc.s = -sc.s;

    float c =  (r.i & 1) != 0 ? sc.s : sc.c;
    c = AS_FLOAT(AS_INT(c) ^ (r.i > 1 ? 0x80000000 : 0));

    if (!FINITE_ONLY_OPT()) {
        c = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : c;
    }

    return c;
}

