/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"


CONSTATTR double
MATH_MANGLE(native_recip)(double x)
{
    // FIXME: Should use IR fdiv with arcp set.
    return __builtin_amdgcn_rcp(x);
}

CONSTATTR double
MATH_MANGLE(native_sqrt)(double x)
{
    return __builtin_sqrt(x);
}

CONSTATTR double
MATH_MANGLE(native_rsqrt)(double x)
{
    return __builtin_amdgcn_rsq(x);
}

