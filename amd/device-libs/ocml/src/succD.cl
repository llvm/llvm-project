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
    long y = AS_LONG(x + 0.0);
    long ix = y + (y >= 0 ? 1l : -1l);
    return BUILTIN_ISNAN_F64(x) || x == PINF_F64 ? x : AS_DOUBLE(ix);
}

