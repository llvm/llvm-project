
#include "mathD.h"

CONSTATTR double
MATH_MANGLE(asin)(double x)
{
    // Computes arcsin(x).
    // The argument is first reduced by noting that arcsin(x)
    // is invalid for abs(x) > 1 and arcsin(-x) = -arcsin(x).
    // For denormal and small arguments arcsin(x) = x to machine
    // accuracy. Remaining argument ranges are handled as follows.
    // For abs(x) <= 0.5 use
    // arcsin(x) = x + x^3*R(x^2)
    // where R(x^2) is a rational minimax approximation to
    // (arcsin(x) - x)/x^3.
    // For abs(x) > 0.5 exploit the identity:
    // arcsin(x) = pi/2 - 2*arcsin(sqrt(1-x)/2)
    // together with the above rational approximation, and
    // reconstruct the terms carefully.

    const double piby2 = 0x1.921fb54442d18p+0;
    const double piby2_tail = 0x1.1a62633145c07p-54;
    const double hpiby2_head = 0x1.921fb54442d18p-1;

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

    // Reconstruct asin carefully in transformed region
    double v;
    if (transform) {
        double s = MATH_FAST_SQRT(r);
        double sh = AS_DOUBLE(AS_ULONG(s) & 0xffffffff00000000UL);
        double st = MATH_FAST_DIV(MATH_MAD(-sh, sh, r), s + sh);
        double p = MATH_MAD(2.0*s, u, -MATH_MAD(-2.0, st, piby2_tail));
        double q = MATH_MAD(-2.0, sh, hpiby2_head);
        v = hpiby2_head - (p - q);
    } else {
        v = MATH_MAD(y, u, y);
    }

    // v = y < 0x1.0p-28 ? y : v;
    if (!FINITE_ONLY_OPT()) {
        v = y > 1.0 ? AS_DOUBLE(QNANBITPATT_DP64) : v;
    }
    v = y == 1.0 ? piby2 : v;

    return BUILTIN_COPYSIGN_F64(v, x);
}

