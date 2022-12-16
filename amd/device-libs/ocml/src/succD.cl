/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(succ)(double x)
{
    long ix = AS_LONG(x);
    long mx = SIGNBIT_DP64 - ix;
    mx = ix < 0 ? mx : ix;
    long t = mx + (x != PINF_F64 && !BUILTIN_ISNAN_F64(x));
    long r = SIGNBIT_DP64 - t;
    r = t < 0 ? r : t;
    r = mx == -1L ? SIGNBIT_DP64 : r;
    return AS_DOUBLE(r);
}

