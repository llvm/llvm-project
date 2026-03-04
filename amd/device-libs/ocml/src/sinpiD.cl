/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

double
MATH_MANGLE(sinpi)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigpired)(ax);
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);

    double s = (r.i & 1) == 0 ? sc.s : sc.c;

    s = AS_DOUBLE(AS_LONG(s) ^ (r.i > 1 ? SIGNBIT_DP64 : 0) ^
                  (AS_LONG(x) ^ AS_LONG(ax)));

    if (!FINITE_ONLY_OPT())
        s = BUILTIN_ISFINITE_F64(x) ? s : QNAN_F64;

    return s;
}

