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

    if (!DAZ_OPT()) {
        ax = BUILTIN_FLDEXP_F32(ax, BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB) ? 24 : 0);
    }

    float z = BUILTIN_EXP2_F32(0x1.555556p-2f * BUILTIN_LOG2_F32(ax));
    z = MATH_MAD(MATH_MAD(MATH_FAST_RCP(z*z), -ax, z), -0x1.555556p-2f, z);

    if (!DAZ_OPT()) {
        z = BUILTIN_FLDEXP_F32(z, BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB) ? -8 : 0);
    }

    z = BUILTIN_CLASS_F32(x, CLASS_QNAN|CLASS_SNAN|CLASS_PINF|CLASS_NINF|CLASS_PZER|CLASS_NZER) ? x : z;
    return BUILTIN_COPYSIGN_F32(z, x);
}

