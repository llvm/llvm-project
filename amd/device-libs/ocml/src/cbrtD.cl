
#include "mathD.h"

CONSTATTR double
MATH_MANGLE(cbrt)(double x)
{
    double a = BUILTIN_ABS_F64(x);
    int e3 = BUILTIN_FREXP_EXP_F64(a);
    int e = (int)BUILTIN_RINT_F32(0x1.555556p-2f * (float)e3);
    a = BUILTIN_FLDEXP_F64(a, -3*e);

    double c = (double)BUILTIN_AMDGPU_EXP2_F32(0x1.555556p-2f * BUILTIN_AMDGPU_LOG2_F32((float)a));
    double c2 = c * c;
    c = MATH_MAD(c, MATH_FAST_DIV(MATH_MAD(-c, c2, a), MATH_MAD(c+c, c2, a)), c);

    c = BUILTIN_FLDEXP_F64(c, e);

    if (!FINITE_ONLY_OPT()) {
        // Is normal or subnormal.
        c = ((x != 0.0) & BUILTIN_ISFINITE_F64(x)) ? c : x;
    }

    return BUILTIN_COPYSIGN_F64(c, x);
}
