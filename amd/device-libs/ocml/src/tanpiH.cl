/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"
#include "trigpiredH.h"

CONSTATTR UGEN(tanpi)

CONSTATTR half
MATH_MANGLE(tanpi)(half x)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F16(x));

    half t = MATH_PRIVATE(tanpired)(r.hi, r.i & (short)1);
    short flip = (((r.i == (short)1) | (r.i == (short)2)) & (r.hi == 0.0h)) ? (short)SIGNBIT_HP16 : (short)0;
    t = AS_HALF((short)((AS_SHORT(t) ^ flip) ^ (AS_SHORT(x) & (short)SIGNBIT_HP16)));

    if (!FINITE_ONLY_OPT()) {
        t = BUILTIN_ISFINITE_F16(x) ? t : QNAN_F16;
    }

    return t;
}

