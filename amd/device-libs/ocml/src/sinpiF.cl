/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

INLINEATTR float
MATH_MANGLE(sinpi)(float x)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;

    float r;
    int i = MATH_PRIVATE(trigpired)(AS_FLOAT(ax), &r);

    float cc;
    float ss = MATH_PRIVATE(sincospired)(r, &cc);

    float s = (i & 1) == 0 ? ss : cc;
    s = AS_FLOAT(AS_INT(s) ^ (i > 1 ? 0x80000000 : 0) ^ (ix ^ ax));

    if (!FINITE_ONLY_OPT()) {
        s = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : s;
    }

    return s;
}

