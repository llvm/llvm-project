
#include "mathD.h"

PUREATTR double
MATH_MANGLE(log1p)(double x)
{
    USE_TABLE(double2, p_tbl, M64_LOGE_EP);

    // Computes natural log(1+x). Algorithm based on:
    // Ping-Tak Peter Tang
    // "Table-driven implementation of the logarithm function in IEEE
    // floating-point arithmetic"
    // ACM Transactions on Mathematical Software (TOMS)
    // Volume 16, Issue 4 (December 1990)
    // Note that we use a lookup table of size 64 rather than 128,
    // and compensate by having extra terms in the minimax polynomial
    // for the kernel approximation.

    // The values exp(-1/16)-1 and exp(1/16)-1
    const double log1p_thresh1 = -0x1.f0540438fd5c3p-5;
    const double log1p_thresh2 =  0x1.082b577d34ed8p-4;
    double ret;

    if (x < log1p_thresh1 | x > log1p_thresh2) {
        ulong ux = as_ulong(1.0 + x);
        int xexp = ((as_int2(ux).hi >> 20) & 0x7ff) - EXPBIAS_DP64;
        double f = as_double(ONEEXPBITS_DP64 | (ux & MANTBITS_DP64));

        int j = as_int2(ux).hi >> 13;
        j = ((0x80 | (j & 0x7e)) >> 1) + (j & 0x1);
        double f1 = (double)j * 0x1.0p-6;
        j -= 64;

        double f2temp = f - f1;
        double m2 = as_double((ulong)(0x3ff - xexp) << EXPSHIFTBITS_DP64);
        double f2l = MATH_MAD(m2, x, m2 - f1);
        double f2g = MATH_MAD(m2, x, -f1) + m2;
        double f2 = xexp <= MANTLENGTH_DP64-1 ? f2l : f2g;
        f2 = (xexp <= -2) | (xexp >= MANTLENGTH_DP64+8) ? f2temp : f2;

        double2 tv = p_tbl[j];
        double z1 = tv.s0;
        double q = tv.s1;

        double u = MATH_FAST_DIV(f2, MATH_MAD(0.5, f2, f1));
        double v = u * u;

        double poly = v * MATH_MAD(v,
                              MATH_MAD(v, 2.23219810758559851206e-03, 1.24999999978138668903e-02),
                              8.33333333333333593622e-02);

        // log2_lead and log2_tail sum to an extra-precise version of log(2)
        const double log2_lead = 0x1.62e42e0000000p-1;
        const double log2_tail = 0x1.efa39ef35793cp-25;

        double z2 = q + MATH_MAD(u, poly, u);
        double dxexp = (double)xexp;
        double r1 = MATH_MAD(dxexp, log2_lead, z1);
        double r2 = MATH_MAD(dxexp, log2_tail, z2);
        ret = r1 + r2;
    } else {
        double r = x;
        double u = MATH_FAST_DIV(r,  2.0 + r);
        double correction = r * u;
        u = u + u;
        double v = u * u;
        double r1 = r;

        double poly = MATH_MAD(v,
                         MATH_MAD(v,
                             MATH_MAD(v, 4.34887777707614552256e-04, 2.23213998791944806202e-03),
                             1.25000000037717509602e-02),
                         8.33333333333317923934e-02);

        double r2 = MATH_MAD(u*v, poly, -correction);

        ret = r1 + r2;
    }

    if (!FINITE_ONLY_OPT()) {
        ret = MATH_MANGLE(isinf)(x) ? x : ret;
        ret = x < -1.0 ? as_double(QNANBITPATT_DP64) : ret;
        ret = x == -1.0 ? as_double(NINFBITPATT_DP64) : ret;
    }

    return ret;
}

