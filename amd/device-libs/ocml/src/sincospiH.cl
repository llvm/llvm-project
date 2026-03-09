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

    short flip = r.i > (short)1 ? (short)SIGNBIT_HP16 : (short)0;
    bool odd = (r.i & (short)1) != (short)0;
    half s = odd ? sc.c : sc.s;

    s = AS_HALF((short)(AS_SHORT(s) ^ (flip ^ AS_SHORT(x) & (short)SIGNBIT_HP16)));

    sc.s = -sc.s;
    half c = AS_HALF((short)(AS_SHORT(odd ? sc.s : sc.c) ^ flip));

    if (!FINITE_ONLY_OPT()) {
        bool finite = BUILTIN_ISFINITE_F16(x);
        c = finite ? c : QNAN_F16;
        s = finite ? s : QNAN_F16;
    }

    *cp = c;
    return s;
}

