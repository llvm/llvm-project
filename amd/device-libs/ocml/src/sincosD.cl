/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigredD.h"

INLINEATTR double
MATH_MANGLE(sincos)(double x, __private double * cp)
{
    double r, rr;
    int regn = MATH_PRIVATE(trigred)(&r, &rr, BUILTIN_ABS_F64(x));

    double cc;
    double ss = MATH_PRIVATE(sincosred2)(r, rr, &cc);

    int flip = regn > 1 ? (int)0x80000000 : 0;
    bool odd = (regn & 1) != 0;

    int2 s = AS_INT2(odd ? cc : ss);
    s.hi ^= flip ^ (AS_INT2(x).hi &(int)0x80000000);
    ss = -ss;
    int2 c = AS_INT2(odd ? ss : cc);
    c.hi ^= flip;

    if (!FINITE_ONLY_OPT()) {
        bool nori = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF);
        s = nori ? AS_INT2(QNANBITPATT_DP64) : s;
        c = nori ? AS_INT2(QNANBITPATT_DP64) : c;
    }

    *cp = AS_DOUBLE(c);
    return AS_DOUBLE(s);
}

