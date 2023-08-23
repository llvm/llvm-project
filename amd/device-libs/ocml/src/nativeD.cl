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

CONSTATTR double
MATH_MANGLE(native_sin)(double x)
{
    return __builtin_sin(x);
}

CONSTATTR double
MATH_MANGLE(native_cos)(double x)
{
    return __builtin_cos(x);
}

CONSTATTR double
MATH_MANGLE(native_exp)(double x)
{
    return __builtin_exp(x);
}

CONSTATTR double
MATH_MANGLE(native_exp2)(double x)
{
    return __builtin_exp2(x);
}

CONSTATTR double
MATH_MANGLE(native_log)(double x)
{
    return __builtin_log(x);
}

CONSTATTR double
MATH_MANGLE(native_log2)(double x)
{
    return __builtin_log2(x);
}

CONSTATTR double
MATH_MANGLE(native_log10)(double x)
{
    return __builtin_log10(x);
}
