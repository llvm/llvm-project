
#include "mathD.h"

// Algorithm: see cbrt

PUREATTR double
MATH_MANGLE(rcbrt)(double x)
{
    USE_TABLE(double, p_inv, M64_CBRT_INV);
    USE_TABLE(double2, p_rcbrt, M64_RCBRT);
    USE_TABLE(double2, p_rem, M64_RCBRT_REM);

    ulong ux = as_ulong(BUILTIN_ABS_F64(x));
    int m = (as_int2(ux).hi >> 20) - EXPBIAS_DP64;

    // Treat subnormals
    ulong uxs = as_ulong(as_double(ONEEXPBITS_DP64 | ux) - 1.0);
    int ms = (as_int2(uxs).hi >> 20) - (2 * EXPBIAS_DP64 - 1);

    bool c = m == -EXPBIAS_DP64;
    ux = c ? uxs : ux;
    m = c ? ms : m;

    int mby3 = m / 3;
    int rem = m - 3*mby3;

    double mf = as_double((ulong)(EXPBIAS_DP64 - mby3) << EXPSHIFTBITS_DP64);

    ux &= MANTBITS_DP64;
    double Y = as_double(HALFEXPBITS_DP64 | ux);

    // nearest integer
    int index = as_int2(ux).hi >> 11;
    index = (0x100 | (index >> 1)) + (index & 1);
    double F = (double)index * 0x1.0p-9;
    
    double f = Y - F;
    double r = f * p_inv[index-256];

    double z = r * MATH_MAD(r,
                       MATH_MAD(r,
                           MATH_MAD(r,
                               MATH_MAD(r,
                                   MATH_MAD(r, 0x1.c67c9ff9c1ce0p-4, -0x1.ff4c33f8fa07cp-4),
                                   0x1.26fabb85cb534p-3),
                               -0x1.61f9add3c0ca4p-3),
                           0x1.c71c71c71c71cp-3),
                       -0x1.5555555555555p-2);

    double2 tv = p_rem[rem+2];
    double Rem_h = tv.s0;
    double Rem_t = tv.s1;

    tv = p_rcbrt[index-256];
    double F_h = tv.s0;
    double F_t = tv.s1;

    double b_h = F_h * Rem_h; 
    double b_t = MATH_MAD(Rem_t, F_h, MATH_MAD(F_t, Rem_h, F_t*Rem_t));

    double ret = MATH_MAD(z, b_h, MATH_MAD(z, b_t, b_t)) + b_h;
    ret *= mf;

    if (!FINITE_ONLY_OPT()) {
        ret = x == 0.0 ? as_double(PINFBITPATT_DP64) : ret;
        ret = BUILTIN_CLASS_F64(x, CLASS_PINF|CLASS_NINF) ? 0.0 : ret;
        ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN) ? x : ret;
    }

    ret = BUILTIN_COPYSIGN_F64(ret, x);

    return ret;
}

