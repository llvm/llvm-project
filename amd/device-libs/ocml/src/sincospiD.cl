/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

double
MATH_MANGLE(sincospi)(double x, __private double * cp)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F64(x));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    int flip = r.i > 1 ? (int)0x80000000 : 0;
    bool odd = (r.i & 1) != 0;

    int2 s = AS_INT2(odd ? sc.c : sc.s);
    s.hi ^= flip ^ (AS_INT2(x).hi & 0x80000000);
    sc.s = -sc.s;
    int2 c = AS_INT2(odd ? sc.s : sc.c);
    c.hi ^= flip;

    if (!FINITE_ONLY_OPT()) {
        bool nori = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF);
        s = nori ? AS_INT2(QNANBITPATT_DP64) : s;
        c = nori ? AS_INT2(QNANBITPATT_DP64) : c;
    }

    *cp = AS_DOUBLE(c);
    return AS_DOUBLE(s);
}

