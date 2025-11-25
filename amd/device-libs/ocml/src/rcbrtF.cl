/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

// Subnormal or zero.
#define IS_LT_SMALLEST_NORMAL(x) (x < 0x1p-126f)

CONSTATTR float
MATH_MANGLE(rcbrt)(float x)
{
    if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
    }

    float ax = BUILTIN_ABS_F32(x);
    bool do_scale = IS_LT_SMALLEST_NORMAL(ax);

    if (!DAZ_OPT()) {
        ax = do_scale ? BUILTIN_FLDEXP_F32(ax, 24) : ax;
    }

    float z = BUILTIN_AMDGPU_EXP2_F32(-0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32(ax));
    z = MATH_MAD(MATH_MAD(z*z, -z*ax, 1.0f), 0x1.555556p-2f*z, z);

    if (!DAZ_OPT()) {
        z = do_scale ? BUILTIN_FLDEXP_F32(z, 8) : z;
    }

    float xi = MATH_FAST_RCP(x);

    // Is normal or subnormal
    z = ((x != 0.0f) & BUILTIN_ISFINITE_F32(x)) ? z : xi;

    return BUILTIN_COPYSIGN_F32(z, x);
}

