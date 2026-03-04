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
MATH_MANGLE(native_exp2)(float x)
{
    // The approximate function expansion of generic exp2 has to
    // handle denormals without DAZ, this does not.
    return __builtin_amdgcn_exp2f(x);
}

CONSTATTR float
MATH_MANGLE(native_exp)(float x)
{
    return MATH_MANGLE(native_exp2)(M_LOG2E_F * x);
}

CONSTATTR float
MATH_MANGLE(native_exp10)(float x)
{
    return MATH_MANGLE(native_exp2)(M_LOG2_10_F * x);
}
