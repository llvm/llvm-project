/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

CONSTATTR float
MATH_MANGLE(sinpi)(float x)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;
    struct redret r = MATH_PRIVATE(trigpired)(AS_FLOAT(ax));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    float s = (r.i & 1) == 0 ? sc.s : sc.c;
    s = AS_FLOAT(AS_INT(s) ^ (r.i > 1 ? 0x80000000 : 0) ^ (ix ^ ax));

    if (!FINITE_ONLY_OPT()) {
        s = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : s;
    }

    return s;
}

