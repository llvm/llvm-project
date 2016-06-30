
#include "mathD.h"

// Algorithm:
//
// x = (2^m)*A
// x = (2^m)*(G+g) with (1 <= G < 2) and (g <= 2^(-8))
// x = (2^m)*2*(G/2+g/2)
// x = (2^m)*2*(F+f) with (0.5 <= F < 1) and (f <= 2^(-9))
//
// Y = (2^(-1))*(2^(-m))*(2^m)*A
// Now, range of Y is: 0.5 <= Y < 1
//
// F = 0x100 + (first 7 mantissa bits) + (8th mantissa bit)
// Now, range of F is: 128 <= F <= 256
// F = F / 256
// Now, range of F is: 0.5 <= F <= 1
//
// f = (Y-F), with (f <= 2^(-9))
//
// cbrt(x) = cbrt(2^m) * cbrt(2) * cbrt(F+f)
// cbrt(x) = cbrt(2^m) * cbrt(2) * cbrt(F) + cbrt(1+(f/F))
// cbrt(x) = cbrt(2^m) * cbrt(2*F) * cbrt(1+r)
//
// r = (f/F), with (r <= 2^(-8))
// r = f*(1/F) with (1/F) precomputed to avoid division
//
// cbrt(x) = cbrt(2^m) * cbrt(G) * (1+poly)
//
// poly = c1*r + c2*(r^2) + c3*(r^3) + c4*(r^4) + c5*(r^5) + c6*(r^6)


PUREATTR double
MATH_MANGLE(cbrt)(double x)
{
    USE_TABLE(double, p_inv, M64_CBRT_INV);
    USE_TABLE(double2, p_cbrt, M64_CBRT);
    USE_TABLE(double2, p_rem, M64_CBRT_REM);

    ulong ux = AS_ULONG(BUILTIN_ABS_F64(x));
    int m = (AS_INT2(ux).hi >> 20) - EXPBIAS_DP64;

    // Treat subnormals
    ulong uxs = AS_ULONG(AS_DOUBLE(ONEEXPBITS_DP64 | ux) - 1.0);
    int ms = (AS_INT2(uxs).hi >> 20) - (2 * EXPBIAS_DP64 - 1);

    bool c = m == -EXPBIAS_DP64;
    ux = c ? uxs : ux;
    m = c ? ms : m;

    int mby3 = m / 3;
    int rem = m - 3*mby3;

    double mf = AS_DOUBLE((ulong)(mby3 + EXPBIAS_DP64) << EXPSHIFTBITS_DP64);

    ux &= MANTBITS_DP64;
    double Y = AS_DOUBLE(HALFEXPBITS_DP64 | ux);

    // nearest integer
    int index = AS_INT2(ux).hi >> 11;
    index = (0x100 | (index >> 1)) + (index & 1);
    double F = (double)index * 0x1.0p-9;
    
    double f = Y - F;
    double r = f * p_inv[index-256];

    double z = r * MATH_MAD(r,
                       MATH_MAD(r,
                           MATH_MAD(r,
                               MATH_MAD(r,
                                   MATH_MAD(r, -0x1.8090d6221a247p-6, 0x1.ee7113506ac13p-6),
                                   -0x1.511e8d2b3183bp-5),
                               0x1.f9add3c0ca458p-5),
                           -0x1.c71c71c71c71cp-4),
                       0x1.5555555555555p-2);

    double2 tv = p_rem[rem+2];
    double Rem_h = tv.s0;
    double Rem_t = tv.s1;

    tv = p_cbrt[index-256];
    double F_h = tv.s0;
    double F_t = tv.s1;

    double b_h = F_h * Rem_h; 
    double b_t = MATH_MAD(Rem_t, F_h, MATH_MAD(F_t, Rem_h, F_t*Rem_t));

    double ret = MATH_MAD(z, b_h, MATH_MAD(z, b_t, b_t)) + b_h;
    ret = BUILTIN_COPYSIGN_F64(ret*mf, x);

    ret = BUILTIN_CLASS_F64(x, CLASS_SNAN|CLASS_QNAN|CLASS_PINF|CLASS_NINF|CLASS_PZER|CLASS_NZER) ? x : ret;

    return ret;
}

