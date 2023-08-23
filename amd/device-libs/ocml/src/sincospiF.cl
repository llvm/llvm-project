/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

float
MATH_MANGLE(sincospi)(float x, __private float *cp)
{
    float ax = BUILTIN_ABS_F32(x);

    struct redret r = MATH_PRIVATE(trigpired)(ax);
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    int flip = r.i > 1 ? 0x80000000 : 0;
    bool odd = (r.i & 1) != 0;
    float s = odd ? sc.c : sc.s;
    s = AS_FLOAT(AS_INT(s) ^ flip ^ (AS_INT(ax) ^ AS_INT(x)));
    sc.s = -sc.s;
    float c = odd ? sc.s : sc.c;
    c = AS_FLOAT(AS_INT(c) ^ flip);

    if (!FINITE_ONLY_OPT()) {
        bool finite = BUILTIN_ISFINITE_F32(ax);
        c = finite ? c : QNAN_F32;
        s = finite ? s : QNAN_F32;
    }

    *cp = c;
    return s;
}

