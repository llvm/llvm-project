/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

CONSTATTR double
MATH_MANGLE(tanpi)(double x)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F64(x));
    int2 t = AS_INT2(MATH_PRIVATE(tanpired)(r.hi, r.i & 1));
    t.hi ^= (((r.i == 1) | (r.i == 2)) & (r.hi == 0.0)) ? 0x80000000 : 0;
    t.hi ^= AS_INT2(x).hi & (int)0x80000000;

    if (!FINITE_ONLY_OPT()) {
        t =  BUILTIN_ISFINITE_F64(x) ? t : AS_INT2(QNANBITPATT_DP64);
    }

    return AS_DOUBLE(t);
}

