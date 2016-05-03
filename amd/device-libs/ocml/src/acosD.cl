
#include "mathD.h"

CONSTATTR double
MATH_MANGLE(acos)(double x)
{
    // Computes arccos(x).
    // The argument is first reduced by noting that arccos(x)
    // is invalid for abs(x) > 1. For denormal and small
    // arguments arccos(x) = pi/2 to machine accuracy.
    // Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arccos(x) = pi/2 - arcsin(x)
    // = pi/2 - (x + x^3*R(x^2))
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arccos(x) = pi - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    const double pi = 0x1.921fb54442d18p+1;
    const double piby2 = 0x1.921fb54442d18p+0;
    const double piby2_head = 0x1.921fb54442d18p+0;
    const double piby2_tail = 0x1.1a62633145c07p-54;

    double y = BUILTIN_ABS_F64(x);
    bool transform = y >= 0.5;

    double rt = 0.5 * (1.0 - y);
    double y2 = y * y;
    double r = transform ? rt : y2;

    // Use a rational approximation for [0.0, 0.5]
    double un = MATH_MAD(r,
                    MATH_MAD(r,
                        MATH_MAD(r,
                            MATH_MAD(r,
                                MATH_MAD(r, 0x1.951665d321061p-15, 0x1.1e5f887a62135p-10),
                                -0x1.c28d390c29690p-5),
                            0x1.1a2bec1b7ef59p-2),
                        -0x1.c7b297e269eacp-2),
                    0x1.d1e4180029834p-3);

    double ud = MATH_MAD(r,
                    MATH_MAD(r,
                        MATH_MAD(r,
                            MATH_MAD(r, 0x1.b1a422982ce76p-4, -0x1.e324ab418f78dp-1),
                            0x1.62021571dccfcp+1),
                        -0x1.a4646f903cdeap+1),
                    0x1.5d6b12001f228p+0);

    double u = r * MATH_FAST_DIV(un, ud);

    // Reconstruct acos carefully in transformed region
    double z;

    if (transform) {
        double s = MATH_FAST_SQRT(r);
        if (x < 0.0) {
            z =  MATH_MAD(-2.0, (s + MATH_MAD(s, u, -piby2_tail)), pi);
        } else {
            // Compute higer precision square root
            double sh = as_double(as_ulong(s) & 0xffffffff00000000UL);
            double st = MATH_FAST_DIV(MATH_MAD(-sh, sh, r), s + sh);
            z = 2.0 * (sh + MATH_MAD(s, u, st));
        }
    } else {
        z = piby2_head - (x - MATH_MAD(-x, u, piby2_tail));
    }

    // z = y < 0x1.0p-56 ? piby2 : z;

    if (!FINITE_ONLY_OPT()) {
        z = y > 1.0 ? as_double(QNANBITPATT_DP64) : z;
    }
    z = x == 1.0 ? 0.0 : z;
    z = x == -1.0 ? pi : z;

    return z;
}


