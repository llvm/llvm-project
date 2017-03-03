/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

INLINEATTR float
MATH_MANGLE(cospi)(float x)
{
    int ax = AS_INT(x) & 0x7fffffff;

    float r;
    int i = MATH_PRIVATE(trigpired)(AS_FLOAT(ax), &r);

    float cc;
    float ss = -MATH_PRIVATE(sincospired)(r, &cc);

    float c =  (i & 1) != 0 ? ss : cc;
    c = AS_FLOAT(AS_INT(c) ^ (i > 1 ? 0x80000000 : 0));

    if (!FINITE_ONLY_OPT()) {
        c = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : c;
    }

    return c;
}

