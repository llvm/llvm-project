/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathH.h"


CONSTATTR half
MATH_MANGLE(native_rcp)(half x)
{
    // FIXME: Should use IR fdiv with arcp set.
    return __llvm_amdgcn_rcp_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_sqrt)(half x)
{
    return __llvm_sqrt_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_rsqrt)(half x)
{
    return __llvm_amdgcn_rsq_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_sin)(half x)
{
    return __llvm_sin_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_cos)(half x)
{
    return __llvm_cos_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_exp)(half x)
{
    return __llvm_exp_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_exp2)(half x)
{
    return __llvm_exp2_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_log)(half x)
{
    return __llvm_log_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_log2)(half x)
{
    return __llvm_log2_f16(x);
}

CONSTATTR half
MATH_MANGLE(native_log10)(half x)
{
    return __llvm_log10_f16(x);

}
