/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

half2
MATH_MANGLE2(sincospi)(half2 x, __private half2 *cp)
{
    half2 s;
    half clo, chi;

    s.lo = MATH_MANGLE(sincospi)(x.lo, &clo);
    s.hi = MATH_MANGLE(sincospi)(x.hi, &chi);
    *cp = (half2)(clo, chi);
    return s;
}

half
MATH_MANGLE(sincospi)(half x, __private half *cp)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    short flip = r.i > (short)1 ? (short)0x8000 : (short)0;
    bool odd = (r.i & (short)1) != (short)0;
    short s = AS_SHORT(odd ? sc.c : sc.s);
    s ^= flip ^ (AS_SHORT(x) & (short)0x8000);
    sc.s = -sc.s;
    short c = AS_SHORT(odd ? sc.s : sc.c);
    c ^= flip;

    if (!FINITE_ONLY_OPT()) {
        bool nori = BUILTIN_CLASS_F16(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF);
        c = nori ? (short)QNANBITPATT_HP16 : c;
        s = nori ? (short)QNANBITPATT_HP16 : s;
    }

    *cp = AS_HALF(c);
    return AS_HALF(s);
}

