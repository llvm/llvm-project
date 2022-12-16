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
    float ax = BUILTIN_ABS_F32(x);
    struct redret r = MATH_PRIVATE(trigpired)(ax);
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    float s = (r.i & 1) == 0 ? sc.s : sc.c;
    s = AS_FLOAT(AS_INT(s) ^ (r.i > 1 ? 0x80000000 : 0) ^ (AS_INT(x) ^ AS_INT(ax)));

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_ISFINITE_F32(ax) ? s : QNAN_F32;
    }

    return s;
}

