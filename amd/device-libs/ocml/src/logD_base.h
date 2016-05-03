
#include "mathD.h"

//   Algorithm:
//
//   Based on:
//   Ping-Tak Peter Tang
//   "Table-driven implementation of the logarithm function in IEEE
//   floating-point arithmetic"
//   ACM Transactions on Mathematical Software (TOMS)
//   Volume 16, Issue 4 (December 1990)
//
//
//   x very close to 1.0 is handled differently, for x everywhere else
//   a brief explanation is given below
//
//   x = (2^m)*A
//   x = (2^m)*(G+g) with (1 <= G < 2) and (g <= 2^(-8))
//   x = (2^m)*2*(G/2+g/2)
//   x = (2^m)*2*(F+f) with (0.5 <= F < 1) and (f <= 2^(-9))
//
//   Y = (2^(-1))*(2^(-m))*(2^m)*A
//   Now, range of Y is: 0.5 <= Y < 1
//
//   F = 0x80 + (first 7 mantissa bits) + (8th mantissa bit)
//   Now, range of F is: 128 <= F <= 256 
//   F = F / 256 
//   Now, range of F is: 0.5 <= F <= 1
//
//   f = -(Y-F), with (f <= 2^(-9))
//
//   log(x) = m*log(2) + log(2) + log(F-f)
//   log(x) = m*log(2) + log(2) + log(F) + log(1-(f/F))
//   log(x) = m*log(2) + log(2*F) + log(1-r)
//
//   r = (f/F), with (r <= 2^(-8))
//   r = f*(1/F) with (1/F) precomputed to avoid division
//
//   log(x) = m*log(2) + log(G) - poly
//
//   log(G) is precomputed
//   poly = (r + (r^2)/2 + (r^3)/3 + (r^4)/4) + (r^5)/5))
//
//   log(2) and log(G) need to be maintained in extra precision
//   to avoid losing precision in the calculations
//
//
//   For x close to 1.0, we employ the following technique to
//   ensure faster convergence.
//
//   log(x) = log((1+s)/(1-s)) = 2*s + (2/3)*s^3 + (2/5)*s^5 + (2/7)*s^7
//   x = ((1+s)/(1-s)) 
//   x = 1 + r
//   s = r/(2+r)

PUREATTR double
#if defined(COMPILING_LOG2)
MATH_MANGLE(log2)(double x)
#elif defined(COMPILING_LOG10)
MATH_MANGLE(log10)(double x)
#else
MATH_MANGLE(log)(double x)
#endif
{
    USE_TABLE(double2, p_tbl, M64_LOGE_EP);

#ifndef COMPILING_LOG2
    // log2_lead and log2_tail sum to an extra-precise version of ln(2)
    const double log2_lead = 0x1.62e42e0000000p-1;
    const double log2_tail = 0x1.efa39ef35793cp-25;
#endif

#if defined(COMPILING_LOG10)
    // log10e_lead and log10e_tail sum to an extra-precision version of log10(e) (19 bits in lead)
    const double log10e_lead = 0x1.bcb7800000000p-2;
    const double log10e_tail = 0x1.8a93728719535p-21;
#elif defined(COMPILING_LOG2)
    // log2e_lead and log2e_tail sum to an extra-precision version of log2(e) (19 bits in lead)
    const double log2e_lead = 0x1.7154400000000p+0;
    const double log2e_tail = 0x1.b295c17f0bbbep-19;
#endif

    double ret;

    if (x < 0x1.e0faap-1 | x > 0x1.1082cp+0) {
        // Deal with subnormal
        double xs = x * 0x1.0p+60;
        int c = x < 0x1.0p-962;
        int expadjust = c ? 60 : 0;
        ulong ux = as_ulong(c ? xs : x);

        int xexp = ((as_int2(ux).hi >> 20) & 0x7ff) - EXPBIAS_DP64 - expadjust;
        double f = as_double(HALFEXPBITS_DP64 | (ux & MANTBITS_DP64));
        int index = as_int2(ux).hi >> 13;
        index = ((0x80 | (index & 0x7e)) >> 1) + (index & 0x1);

        double2 tv = p_tbl[index - 64];
        double z1 = tv.s0;
        double q = tv.s1;

        double f1 = index * 0x1.0p-7;
        double f2 = f - f1;
        double u = MATH_FAST_DIV(f2, MATH_MAD(f2, 0.5, f1));
        double v = u * u;
        double poly = v * MATH_MAD(v,
                              MATH_MAD(v, 0x1.249423bd94741p-9, 0x1.9999999865edep-7),
                              0x1.5555555555557p-4);
        double z2 = q + MATH_MAD(u, poly, u);

        double dxexp = (double)xexp;
#if defined (COMPILING_LOG10)
        // Add xexp * log(2) to z1,z2 to get log(x)
        double r1 = MATH_MAD(dxexp, log2_lead, z1);
        double r2 = MATH_MAD(dxexp, log2_tail, z2);
        ret = MATH_MAD(log10e_lead, r1, MATH_MAD(log10e_lead, r2, MATH_MAD(log10e_tail, r1, log10e_tail*r2)));
#elif defined(COMPILING_LOG2)
        double r1 = MATH_MAD(log2e_lead, z1, dxexp);
        double r2 = MATH_MAD(log2e_lead, z2, MATH_MAD(log2e_tail, z1, log2e_tail*z2));
        ret = r1 + r2;
#else
        double r1 = MATH_MAD(dxexp, log2_lead, z1);
        double r2 = MATH_MAD(dxexp, log2_tail, z2);
        ret = r1 + r2;
#endif
    } else {
        double r = x - 1.0;
        double u = MATH_FAST_DIV(r, 2.0 + r);
        double correction = r * u;
        u = u + u;
        double v = u * u;
        double r1 = r;
        double r2 = MATH_MAD(u*v, MATH_MAD(v,
                                      MATH_MAD(v,
                                          MATH_MAD(v, 0x1.c8034c85dfff0p-12, 0x1.2492307f1519fp-9),
                                          0x1.9999999bac6d4p-7),
                                      0x1.55555555554e6p-4), -correction);

#if defined(COMPILING_LOG10)
        r = r1;
        r1 = as_double(as_ulong(r1) & 0xffffffff00000000UL);
        r2 = r2 + (r - r1);
        ret = MATH_MAD(log10e_lead, r1, MATH_MAD(log10e_lead, r2, MATH_MAD(log10e_tail, r1, log10e_tail * r2)));
#elif defined(COMPILING_LOG2)
        r = r1;
        r1 = as_double(as_ulong(r1) & 0xffffffff00000000UL);
        r2 = r2 + (r - r1);
        ret = MATH_MAD(log2e_lead, r1, MATH_MAD(log2e_lead, r2, MATH_MAD(log2e_tail, r1, log2e_tail*r2)));
#else
        ret = r1 + r2;
#endif
    }

    if (!FINITE_ONLY_OPT()) {
        ret = MATH_MANGLE(isinf)(x) ? x : ret;
        ret = MATH_MANGLE(isnan)(x) | (x < 0.0) ? as_double(QNANBITPATT_DP64) : ret;
        ret = x == 0.0 ? as_double(NINFBITPATT_DP64) : ret;
    }

    return ret;
}

