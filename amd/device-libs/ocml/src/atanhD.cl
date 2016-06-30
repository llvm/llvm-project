
#include "mathD.h"

PUREATTR double
MATH_MANGLE(atanh)(double x)
{
    double ax = BUILTIN_ABS_F64(x);
    double ret;

    if (ax < 0.5) {
        double x2 = x * x;

        double pn = MATH_MAD(x2,
                        MATH_MAD(x2,
                            MATH_MAD(x2,
                                MATH_MAD(x2,
                                    MATH_MAD(x2, -0x1.b711000f5a53bp-14, 0x1.d6b0a4cfde8fcp-6),
                                    -0x1.2090bb7302592p-2),
                                0x1.c4f4f6baa48ffp-1),
                            -0x1.1a53706989746p+0),
                        0x1.e638b7bbea45ep-2);

        double pd = MATH_MAD(x2,
                        MATH_MAD(x2,
                            MATH_MAD(x2,
                                MATH_MAD(x2,
                                    MATH_MAD(x2, -0x1.25c7216683ecap-5, 0x1.fb81b3fe42b33p-2),
                                    -0x1.2164ca4f0c6f3p+1),
                                0x1.22a7720caaa5dp+2),
                            -0x1.0a71c2944b0bfp+2),
                        0x1.6caa89ccefb46p+0);

        ret = MATH_MAD(x, x2*MATH_FAST_DIV(pn, pd), x);
    } else {
        // |x| >= 0.5
        // Note that atanh(x) = 0.5 * ln((1+x)/(1-x))
        // For greater accuracy we use
        // ln((1+x)/(1-x)) = ln(1 + 2x/(1-x)) = log1p(2x/(1-x)).
        double r = 0.5 * MATH_MANGLE(log1p)(MATH_FAST_DIV(2.0*ax, 1.0 - ax));

        if (!FINITE_ONLY_OPT()) {
            ret = ax == 1.0 ? AS_DOUBLE(PINFBITPATT_DP64) : AS_DOUBLE(QNANBITPATT_DP64);
            ret = ax < 1.0 ? r : ret;
        } else {
	    ret = r;
        }

        ret = BUILTIN_COPYSIGN_F64(ret, x);
    }


    return ret;
}

