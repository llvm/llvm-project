/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

UGEN(sinpi)

half
MATH_MANGLE(sinpi)(half x)
{
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F16(x) ? QNAN_F16 : x;

    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x));
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    half s = (r.i & (short)1) == (short)0 ? sc.s : sc.c;
    short flip = r.i > (short)1 ? (short)SIGNBIT_HP16 : (short)0;

    s = AS_HALF((short)(AS_SHORT(s) ^ (flip ^ (AS_SHORT(x) & (short)SIGNBIT_HP16))));

    return s;
}

