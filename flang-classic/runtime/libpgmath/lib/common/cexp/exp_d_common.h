
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "math_common.h"
#include "sleef_common.h"

static double const L2E         = 1.4426950408889634e+0;
static double const LN2_HI      = 6.9314718055994529e-1;
static double const LN2_LO      = 2.3190468138462996e-17;
static double const EXP_D_OVF   = 0x1.6d49df5728ea2p10; //((2048.0 + 60.0) * LN2_HI);
static double const EXP_POLY_11 = 2.5022322536502990E-008;
static double const EXP_POLY_10 = 2.7630903488173108E-007;
static double const EXP_POLY_9  = 2.7557514545882439E-006;
static double const EXP_POLY_8  = 2.4801491039099165E-005;
static double const EXP_POLY_7  = 1.9841269589115497E-004;
static double const EXP_POLY_6  = 1.3888888945916380E-003;
static double const EXP_POLY_5  = 8.3333333334550432E-003;
static double const EXP_POLY_4  = 4.1666666666519754E-002;
static double const EXP_POLY_3  = 1.6666666666666477E-001;
static double const EXP_POLY_2  = 5.0000000000000122E-001;
static double const EXP_POLY_1  = 1.0000000000000000E+000;
static double const EXP_POLY_0  = 1.0000000000000000E+000;
static double const DBL2INT_CVT = 0x1.8p52;

static void __exp_d_scalar_kernel(double a, double *poly, long long int *scale)
{
    if ( a >  EXP_D_OVF ) a =  EXP_D_OVF;
    if ( a < -EXP_D_OVF ) a = -EXP_D_OVF;

    // calculating exponent; stored in the LO of each 64-bit block
    unsigned long long int i = D2L(fma(a, L2E, DBL2INT_CVT));

    // calculate mantissa
    // fast mul rint
    double t = fma(a, L2E, DBL2INT_CVT) - DBL2INT_CVT;
    double m = fma(t, -LN2_HI, a);
           m = fma(t, -LN2_LO, m);

    // evaluate highest 8 terms of polynomial with estrin, then switch to horner
    double z10 = fma(EXP_POLY_11, m, EXP_POLY_10);
    double z8  = fma(EXP_POLY_9, m, EXP_POLY_8);
    double z6  = fma(EXP_POLY_7, m, EXP_POLY_6);
    double z4  = fma(EXP_POLY_5, m, EXP_POLY_4);

    double m2 = m * m;
    z8 = fma(z10, m2, z8);
    z4 = fma(z6, m2, z4);

    double m4 = m2 * m2;
    z4 = fma(z8, m4, z4);

    t = fma(z4, m, EXP_POLY_3);
    t = fma(t, m, EXP_POLY_2);
    t = fma(t, m, EXP_POLY_1);
    t = fma(t, m, EXP_POLY_0);

    *poly  = t * 0x1.p512;
    *scale = ((signed long long int)(i << 13) >> 13) - 512;
}

static vdouble INLINE __vexp_d_poly(vdouble m)
{
    vdouble const exp_poly_11 = vSETd(EXP_POLY_11);
    vdouble const exp_poly_10 = vSETd(EXP_POLY_10);
    vdouble const exp_poly_9  = vSETd(EXP_POLY_9 );
    vdouble const exp_poly_8  = vSETd(EXP_POLY_8 );
    vdouble const exp_poly_7  = vSETd(EXP_POLY_7 );
    vdouble const exp_poly_6  = vSETd(EXP_POLY_6 );
    vdouble const exp_poly_5  = vSETd(EXP_POLY_5 );
    vdouble const exp_poly_4  = vSETd(EXP_POLY_4 );
    vdouble const exp_poly_3  = vSETd(EXP_POLY_3 );
    vdouble const exp_poly_2  = vSETd(EXP_POLY_2 );
    vdouble const exp_poly_1  = vSETd(EXP_POLY_1 );
    vdouble const exp_poly_0  = vSETd(EXP_POLY_0 );

    // evaluate highest 8 terms of polynomial with estrin, then switch to horner
    vdouble z10 = vfma_vd_vd_vd_vd(exp_poly_11, m, exp_poly_10);
    vdouble z8  = vfma_vd_vd_vd_vd(exp_poly_9, m, exp_poly_8);
    vdouble z6  = vfma_vd_vd_vd_vd(exp_poly_7, m, exp_poly_6);
    vdouble z4  = vfma_vd_vd_vd_vd(exp_poly_5, m, exp_poly_4);

    vdouble m2 = vmul_vd_vd_vd(m, m);
    z8 = vfma_vd_vd_vd_vd(z10, m2, z8);
    z4 = vfma_vd_vd_vd_vd(z6, m2, z4);

    vdouble m4 = vmul_vd_vd_vd(m2, m2);
    z4 = vfma_vd_vd_vd_vd(z8, m4, z4);

    vdouble t = vfma_vd_vd_vd_vd(z4, m, exp_poly_3);
    t = vfma_vd_vd_vd_vd(t, m, exp_poly_2);
    t = vfma_vd_vd_vd_vd(t, m, exp_poly_1);
    t = vfma_vd_vd_vd_vd(t, m, exp_poly_0);
    return t;
}

static void INLINE
__vexp_d_kernel(vdouble vx, vdouble * vpoly, vdouble * vscale)
{
    // This algorithm computes exp(vx) in a form of
    // 2^(vscale) * vpoly, unevaluated. vscale is an integer.
    // The intended use of this form is for subsequent
    // multiplication of vpoly by sin/cos, which can be small.
    // We don't know the values of sin/cos apriori, so need
    // to compute exp() with the extended range, thus the need
    // to hold the scale bits in a separate integer, wider than
    // 8 bits provided by the IEEE binary32 format.
    // To avoid potential loss of accuracy in denormals we
    // make sure that vpoly * sin() is a normal number - for
    // that we shift some of the scaling from vscale to vpoly.
    // Later scaling by 2^(vscale) may still result in a denormal
    // and the loss of accuracy, but in this case it will be
    // bound by ~1 ulp, which is tolerable for the implementation.

    // exp algorithm outline: we reduce argument to +-ln2/2 interval
    // by representing x = N*ln2 + z, in this case exp(x) = 2^N * exp(z).
    // We want N to be integer and thus it is obtained as:
    //    a) N = round_to_nearest_int(x * 1/ln2)
    // And reduced argument z is:
    //    b) z = x - N*ln2
    // We need to guarantee that |z| < ln2/2 and we need to estimate
    // the error introduced by reduction too.
    // exp(x) can quickly over/underflow, so given the bounds on argument
    // x in which we want to compute exp(), we can infer the bounds on
    // N and decide on the precision needs in finite approximations
    // of 1/ln2 and ln2 constants.
    //
    // Once the reduced argument z is known, we compute the exp(z) as
    // a polynomial approximation. We would like exp(0) to
    // be exactly 1, so we chose the polynomial coefficients accordingly.

    // FIXME: compute overflow threshold more accurately

    // clamp range of x to over/underflow bounds to avoid errors in
    // range reduction procedure leading to unbounded polynomial.
    vopmask mover   = vgt_vo_vd_vd(vx, vSETd(EXP_D_OVF));
    vx = vsel_vd_vo_vd_vd(mover, vSETd(EXP_D_OVF), vx);
    // exp underflows for x <= EXP_LO, it cannot be recovered with later
    // multiply by sincos, which is <= 1
    vopmask munder  = vle_vo_vd_vd(vx, vSETd(-EXP_D_OVF));
    vx = vsel_vd_vo_vd_vd(munder, vSETd(-EXP_D_OVF), vx);
    // tt = [x / ln2] = [x * log2(e)], convert to integer using right-shifter
    const vdouble vRS = vSETd(DBL2INT_CVT);
    // least significant bits of t now contain an integer rounded
    // according to current rounding mode, default: to nearest
    // TODO: this algorithm will fail in directed rounding mode because
    // of the over/underestimate in t and thus tt.
    vdouble t = vfma_vd_vd_vd_vd(vx, vSETd(L2E), vRS);                          PRINT(t);
    // subtract right-shifter to obtain the integer as a normalized FP number
    vdouble tt= vsub_vd_vd_vd(t, vRS);                                          PRINT(tt);

    // FMA is essential here. If no FMA, need to provide exact multiplication by
    // LN2_HI, and this constant shall be changed to have e.g. 10 trailing zeros
    // so that the product can absorb the 10 bits of tt.
    vdouble z = vfma_vd_vd_vd_vd(tt, vSETd(-LN2_HI), vx);                       PRINT(z);
            z = vfma_vd_vd_vd_vd(tt, vSETd(-LN2_LO), z);                        PRINT(z);

    vint2 exponent = vD2L(t);                                                   PRINT(exponent);
          // sign-extend integer exp:
          // wipe dummy FP sign, FP exponent field and two leading FP mantissa
          // bits (1 bit implicit), which are leftovers from right-shifter.
          // NOTE: this 64-bit code works even though we use 32-bit SIMD shifts
          exponent = vsll_vi2_vi2_i(exponent, 13);                              PRINT(exponent);
          exponent = vsra_vi2_vi2_i(exponent, 13);                              PRINT(exponent);

    // compute polynomial approximation, it shall be on the order
    // of exp() in the reduced range [-ln2/2, ln2/2], so zz is
    // in [1/sqrt(2), sqrt(2)] or in (1/2, 2)
    vdouble zz = __vexp_d_poly(z);                                              PRINT(zz);

    // exponent scaling factor is now somewhere in [-1024 - 53; 1024 + 1024 + 53].
    // Subtract 512 from scaling and add it back to polynomial so that later
    // polynomial * sin() always results in normalized numbers.
    // Plus it also makes exponent range symmetric:
    // [-1024 - 53 - 512; 1024 + 53 + 512], only 12 bits of
    // storage together with the sign.
    exponent = vsub64_vi2_vi2_vi2(exponent, vSETll(512));                       PRINT(exponent);
    // new poly is in (2^511, 2^513)
    zz = vmul_vd_vd_vd(zz, vL2D(vSETll((512ULL+DB_EXP_BIAS) << (DB_PREC_BITS-1)))); PRINT(zz);

    *vpoly = zz;
    *vscale = vL2D(exponent);
    return;
}
