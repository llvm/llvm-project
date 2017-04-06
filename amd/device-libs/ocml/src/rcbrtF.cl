/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathF.h"

CONSTATTR INLINEATTR float
MATH_MANGLE(rcbrt)(float x)
{
    if (DAZ_OPT()) {
        x = BUILTIN_CANONICALIZE_F32(x);
    }

    float ax = BUILTIN_ABS_F32(x);
    
    if (!DAZ_OPT()) {
        ax = BUILTIN_FLDEXP_F32(ax, BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB) ? 24 : 0);
    }

    float z = BUILTIN_EXP2_F32(-0x1.555556p-2f * BUILTIN_LOG2_F32(ax));
    z = MATH_MAD(MATH_MAD(z*z, -z*ax, 1.0f), 0x1.555556p-2f*z, z);

    if (!DAZ_OPT()) {
        z = BUILTIN_FLDEXP_F32(z, BUILTIN_CLASS_F32(x, CLASS_NSUB|CLASS_PSUB) ? 8 : 0);
    }

    float xi = MATH_FAST_RCP(x);
    z = BUILTIN_CLASS_F32(x, CLASS_SNAN|CLASS_QNAN|CLASS_PZER|CLASS_NZER|CLASS_PINF|CLASS_NINF) ? xi : z;

    return BUILTIN_COPYSIGN_F32(z, x);
}

