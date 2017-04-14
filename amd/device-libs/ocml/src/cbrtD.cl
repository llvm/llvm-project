
#include "mathD.h"

CONSTATTR INLINEATTR double
MATH_MANGLE(cbrt)(double x)
{
    double a = BUILTIN_ABS_F64(x);
    int e3 = BUILTIN_FREXP_EXP_F64(a);
    int e = (int)BUILTIN_RINT_F32(0x1.555556p-2f * (float)e3);
    a = BUILTIN_FLDEXP_F64(a, -3*e);

    double c = (double)BUILTIN_EXP2_F32(0x1.555556p-2f * BUILTIN_LOG2_F32((float)a));
    double c2 = c * c;
    c = MATH_MAD(c, MATH_FAST_DIV(MATH_MAD(-c, c2, a), MATH_MAD(c+c, c2, a)), c);

    c = BUILTIN_FLDEXP_F64(c, e);

    if (!FINITE_ONLY_OPT()) {
        c = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_PINF|CLASS_NINF|CLASS_PZER|CLASS_NZER) ? x : c;
    }

    return BUILTIN_COPYSIGN_F64(c, x);
}
