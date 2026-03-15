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
    if (!FINITE_ONLY_OPT())
        x = BUILTIN_ISINF_F64(x) ? QNAN_F64 : x;

    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigpired)(ax);
    double t = MATH_PRIVATE(tanpired)(r.hi, r.i & 1);

    long flip = (((r.i == 1) | (r.i == 2)) & (r.hi == 0.0)) ? SIGNBIT_DP64 : 0;

    return AS_DOUBLE((AS_LONG(t) ^ flip) ^ (AS_LONG(x) & SIGNBIT_DP64));
}

