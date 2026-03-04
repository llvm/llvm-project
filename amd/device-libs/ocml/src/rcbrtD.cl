/*===--------------------------------------------------------------------------
 *                   ROCm Device Libraries
 *
 * This file is distributed under the University of Illinois Open Source
 * License. See LICENSE.TXT for details.
 *===------------------------------------------------------------------------*/

#include "mathD.h"

CONSTATTR double
MATH_MANGLE(rcbrt)(double x)
{
    double a = BUILTIN_ABS_F64(x);
    int e3 = BUILTIN_FREXP_EXP_F64(a);
    int e = (int)BUILTIN_RINT_F32(0x1.555556p-2f * (float)e3);
    a = BUILTIN_FLDEXP_F64(a, -3*e);

    double c = (double)BUILTIN_AMDGPU_EXP2_F32(-0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32((float)a));

    // Correction is c + c*(1 - a c^3)/(1 + 2 a c^3)
    //  = c + c*t/(3 - 2t) where t = 1 - a c^3
    // use t/(3 - 2t) ~ t/3 + 2 t^2 / 9 + 4 t^3 / 27 ...
    // compute t with extra precision for better accuracy
    double c3 = c * c * c;
    double t = MATH_MAD(-a, c3, 1.0);
    c = MATH_MAD(c, t*MATH_MAD(t, 0x1.c71c71c71c8b2p-3, 0x1.5555555555685p-2), c);

    c = BUILTIN_FLDEXP_F64(c, -e);

    if (!FINITE_ONLY_OPT()) {
        c = a == PINF_F64 ? 0.0 : c;
        c = x == 0.0 ? PINF_F64 : c;
    }

    return BUILTIN_COPYSIGN_F64(c, x);
}

