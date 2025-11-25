/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"
#include "trigpiredD.h"

double
MATH_MANGLE(cospi)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    struct redret r = MATH_PRIVATE(trigpired)(ax);
    struct scret sc = MATH_PRIVATE(sincospired)(r.hi);
    sc.s = -sc.s;

    double c = (r.i & 1) == 0 ? sc.c : sc.s;
    c = r.i > 1 ? -c : c;

    if (!FINITE_ONLY_OPT() && !BUILTIN_ISFINITE_F64(ax)) {
        c = QNAN_F64;
    }

    return c;
}

