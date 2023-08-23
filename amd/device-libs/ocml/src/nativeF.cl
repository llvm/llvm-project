/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"


CONSTATTR float
MATH_MANGLE(native_recip)(float x)
{
    // FIXME: Should use IR fdiv with arcp set.
    return __builtin_amdgcn_rcpf(x);
}

CONSTATTR float
MATH_MANGLE(native_sqrt)(float x)
{
    return __builtin_sqrtf(x);
}

CONSTATTR float
MATH_MANGLE(native_rsqrt)(float x)
{
    return __builtin_amdgcn_rsqf(x);
}

CONSTATTR float
MATH_MANGLE(native_sin)(float x) {
    return __builtin_sinf(x);
}

CONSTATTR float
MATH_MANGLE(native_cos)(float x)
{
    return __builtin_cosf(x);
}
