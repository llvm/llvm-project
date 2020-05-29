/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

REQUIRES_16BIT_INSTS half2
MATH_MANGLE2(sincos)(half2 x, __private half2 *cp)
{
    half2 s;
    half clo, chi;
    s.lo = MATH_MANGLE(sincos)(x.lo, &clo);
    s.hi = MATH_MANGLE(sincos)(x.hi, &chi);
    *cp = (half2)(clo, chi);
    return s;
}

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(sincos)(half x, __private half *cp)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);

    short flip = r.i > (short)1 ? (short)0x8000 : (short)0;
    bool odd = (r.i & (short)1) != (short)0;
    short s = AS_SHORT(odd ? sc.c : sc.s);
    s ^= flip ^ (AS_SHORT(x) & (short)0x8000);
    sc.s = -sc.s;
    short c = AS_SHORT(odd ? sc.s : sc.c);
    c ^= flip;

    if (!FINITE_ONLY_OPT()) {
        bool finite = BUILTIN_ISFINITE_F16(ax);
        c = finite ? c : (short)QNANBITPATT_HP16;
        s = finite ? s : (short)QNANBITPATT_HP16;
    }

    *cp = AS_HALF(c);
    return AS_HALF(s);
}

