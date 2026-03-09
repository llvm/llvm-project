/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigredH.h"

half2
MATH_MANGLE2(sincos)(half2 x, __private half2 *cp)
{
    half2 s;
    half clo, chi;
    s.lo = MATH_MANGLE(sincos)(x.lo, &clo);
    s.hi = MATH_MANGLE(sincos)(x.hi, &chi);
    *cp = (half2)(clo, chi);
    return s;
}

CONSTATTR half
MATH_MANGLE(sincos)(half x, __private half *cp)
{
    half ax = BUILTIN_ABS_F16(x);
    struct redret r = MATH_PRIVATE(trigred)(ax);
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);

    short flip = r.i > (short)1 ? (short)SIGNBIT_HP16 : (short)0;
    bool odd = (r.i & (short)1) != (short)0;
    half s = odd ? sc.c : sc.s;

    s = AS_HALF((short)(AS_SHORT(s) ^ (flip ^ AS_SHORT(x) & (short)SIGNBIT_HP16)));

    sc.s = -sc.s;
    half c = odd ? sc.s : sc.c;
    c = AS_HALF((short)(AS_SHORT(c) ^ flip));

    if (!FINITE_ONLY_OPT()) {
        bool finite = BUILTIN_ISFINITE_F16(ax);
        c = finite ? c : QNAN_F16;
        s = finite ? s : QNAN_F16;
    }

    *cp = c;
    return s;
}

