/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

INLINEATTR float
MATH_MANGLE(sincospi)(float x, __private float *cp)
{
    int ix = AS_INT(x);
    int ax = ix & 0x7fffffff;

    float t;
    int i = MATH_PRIVATE(trigpired)(AS_FLOAT(ax), &t);

    float cc;
    float ss = MATH_PRIVATE(sincospired)(t, &cc);

    int flip = i > 1 ? 0x80000000 : 0;
    bool odd = (i & 1) != 0;
    float s = odd ? cc : ss;
    s = AS_FLOAT(AS_INT(s) ^ flip ^ (ax ^ ix));
    ss = -ss;
    float c = odd ? ss : cc;
    c = AS_FLOAT(AS_INT(c) ^ flip);

    if (!FINITE_ONLY_OPT()) {
        c = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : c;
        s = ax >= PINFBITPATT_SP32 ? AS_FLOAT(QNANBITPATT_SP32) : s;
    }

    *cp = c;
    return s;
}

