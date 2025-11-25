/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR float
MATH_MANGLE(cbrt)(float x)
{
    if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
    }

    float ax = BUILTIN_ABS_F32(x);
    bool denorm_or_zero = ax < 0x1p-126f;

    if (!DAZ_OPT()) {
        ax = denorm_or_zero ?
             BUILTIN_FLDEXP_F32(ax, 24) : ax;
    }

    float z = BUILTIN_AMDGPU_EXP2_F32(0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32(ax));
    z = MATH_MAD(MATH_MAD(MATH_FAST_RCP(z*z), -ax, z), -0x1.555556p-2f, z);

    if (!DAZ_OPT()) {
        z = denorm_or_zero ?
            BUILTIN_FLDEXP_F32(z, -8) : z;
    }

    // Is normal or subnormal.
    z = ((x != 0.0f) & BUILTIN_ISFINITE_F32(x)) ? z : x;
    return BUILTIN_COPYSIGN_F32(z, x);
}

