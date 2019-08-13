/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"


REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(native_rcp)(half x)
{
    // FIXME: Should use IR fdiv with arcp set.
    return __builtin_amdgcn_rcph(x);
}

CONSTATTR half
MATH_MANGLE(native_sqrt)(half x)
{
    return __builtin_sqrtf16(x);
}

REQUIRES_16BIT_INSTS CONSTATTR half
MATH_MANGLE(native_rsqrt)(half x)
{
    return __builtin_amdgcn_rsqh(x);
}

CONSTATTR half
MATH_MANGLE(native_sin)(half x)
{
    return __builtin_sinf16(x);
}

CONSTATTR half
MATH_MANGLE(native_cos)(half x)
{
    return __builtin_cosf16(x);
}

CONSTATTR half
MATH_MANGLE(native_exp)(half x)
{
    return __builtin_expf16(x);
}

CONSTATTR half
MATH_MANGLE(native_exp2)(half x)
{
    return __builtin_exp2f16(x);
}

CONSTATTR half
MATH_MANGLE(native_log)(half x)
{
    return __builtin_logf16(x);
}

CONSTATTR half
MATH_MANGLE(native_log2)(half x)
{
    return __builtin_log2f16(x);
}

CONSTATTR half
MATH_MANGLE(native_log10)(half x)
{
    return __builtin_log10f16(x);

}
