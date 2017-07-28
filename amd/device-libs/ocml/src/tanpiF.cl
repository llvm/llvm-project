/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigpiredF.h"

CONSTATTR float
MATH_MANGLE(tanpi)(float x)
{
    struct redret r = MATH_PRIVATE(trigpired)(BUILTIN_ABS_F32(x));
    int t = AS_INT(MATH_PRIVATE(tanpired)(r.hi, r.i & 1));
    t ^= (((r.i == 1) | (r.i == 2)) & (r.hi == 0.0f)) ? (int)0x80000000 : 0;
    t ^= AS_INT(x) & (int)0x80000000;

    if (!FINITE_ONLY_OPT()) {
        t =  BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN|CLASS_NINF|CLASS_PINF) ? QNANBITPATT_SP32 : t;
    }

    return AS_FLOAT(t);
}

