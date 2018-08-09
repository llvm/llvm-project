/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

// Value of log2(10)
#define M_LOG2_10_F 0x1.a934f0p+1f


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

CONSTATTR float
MATH_MANGLE(native_exp)(float x)
{
    return __builtin_expf(x);
}

CONSTATTR float
MATH_MANGLE(native_exp2)(float x)
{
    return __builtin_exp2f(x);
}

CONSTATTR float
MATH_MANGLE(native_exp10)(float x)
{
    return __builtin_exp2f(M_LOG2_10_F * x);
}

CONSTATTR float
MATH_MANGLE(native_log)(float x)
{
    return __builtin_logf(x);
}

CONSTATTR float
MATH_MANGLE(native_log2)(float x)
{
    return __builtin_log2f(x);
}

CONSTATTR float
MATH_MANGLE(native_log10)(float x)
{
    return __builtin_log10f(x);
}

