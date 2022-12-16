/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"
#include "trigredF.h"

float
MATH_MANGLE(sin)(float x)
{
    float ax = BUILTIN_ABS_F32(x);

    struct redret r = MATH_PRIVATE(trigred)(ax);

#if defined EXTRA_PRECISION
    struct scret sc = MATH_PRIVATE(sincosred2)(r.hi, r.lo);
#else
    struct scret sc = MATH_PRIVATE(sincosred)(r.hi);
#endif

    float s = (r.i & 1) != 0 ? sc.c : sc.s;
    s = AS_FLOAT(AS_INT(s) ^ (r.i > 1 ? 0x80000000 : 0) ^
                 (AS_INT(x) ^ AS_INT(ax)));

    if (!FINITE_ONLY_OPT()) {
        s = BUILTIN_ISFINITE_F32(ax) ? s : QNAN_F32;
    }

    return s;
}

