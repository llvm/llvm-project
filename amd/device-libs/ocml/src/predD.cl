/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(pred)(double x)
{
    long ix = AS_LONG(x) + (x > 0.0 ? -1L : 1L);
    double y = x == 0.0 ? -0x1p-1074 : AS_DOUBLE(ix);
    return BUILTIN_ISNAN_F64(x) || x == NINF_F64 ? x : y;
}

