
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "math_common.h"
#include "sleef_common.h"

#define L2E          1.4426950408889634e+0f
#define LN2_0        0x1.62E43p-01
#define LN2_1       -0x1.05C61p-29

// 2^(-27), 0.99 ulp
#define EXP_C0       I2F(0x3f800000)
#define EXP_C1       I2F(0x3f800000)
#define EXP_C2       I2F(0x3effffff)
#define EXP_C3       I2F(0x3e2aaa57)
#define EXP_C4       I2F(0x3d2aab8d)
#define EXP_C5       I2F(0x3c091474)
#define EXP_C6       I2F(0x3ab5b798)

static INLINE
float __exp_poly(float z)
{
#if (defined EXP_C7)
    float zz =             EXP_C7;
          zz = fmaf(zz, z, EXP_C6);
          zz = fmaf(zz, z, EXP_C5);
#elif (defined EXP_C6)
    float zz =             EXP_C6;
          zz = fmaf(zz, z, EXP_C5);
#else
    float zz =             EXP_C5;
#endif
          zz = fmaf(zz, z, EXP_C4);
          zz = fmaf(zz, z, EXP_C3);
          zz = fmaf(zz, z, EXP_C2);
          zz = fmaf(zz, z, EXP_C1);
#if !defined EXP_C0_lo
          zz = fmaf(zz, z, EXP_C0);
#else
          zz = fmaf(zz, z, EXP_C0_lo);
          zz = zz + EXP_C0;
#endif
    return zz;
}

static INLINE void __exp_scalar_kernel(float a, float * poly, int * scale)
{
    *scale = 0;

    // clamp range of x to over/underflow bounds to avoid errors in
    // range reduction procedure leading to unbounded polynomial.
    // NOTE: we let the NaNs fall through
    // a >= 3.0f*EXP_HI overflow cannot be recovered by sin/cos

    #define EXP_HI (128.0f * logf(2.0f))
    if ( a >= 3.0f*EXP_HI )
    {
        a = 3.0f*EXP_HI;
    }
    if (a <= -3.0f*EXP_HI) //underflow, cannot recover with sincos.
    {
        a = -3.0f*EXP_HI;
    }

    float t = fmaf(a, L2E, 0x1.8p23);
    float tt = t - 0x1.8p23;
    // FMA is essential here. If no FMA, need to provide exact multiplication by
    // LN2_0, and this constant shall be changed to have 10 trailing zeros
    // so that the product can absorb the 10 bits of tt.
    float z = fmaf(tt, -LN2_0, a);
          z = fmaf(tt, -LN2_1, z);

    int exp = F2I(t);
    // sign-extend integer exp:
    // wipe dummy FP sign, FP exponent field and two leading FP mantissa bits
    // (1 bit implicit), which are leftovers from right-shifter
    exp <<= 10;
    exp >>= 10;

    // compute polynomial approximation, it shall be on the order
    // of exp() in the reduced range [-ln2/2, ln2/2], so zz is in [1/sqrt(2), sqrt(2)] in (1/2, 2)
    float zz = __exp_poly(z);

    // exp scaling factor is now somewhere in [-128 - 24; 128 + 128 + 24]
    // Subtract 64 in order to
    // compensate for denormals in sin(), also make exp range symmetric.
    // Adjust poly accordingly too.
    exp -= 64;
    // Here we multiply by a FP constant instead of integer addition to exp bits
    // - to preserve the NaNs
    *poly = zz * I2F((64 + 127) << 23);
    // Now exp is in [-128 - 24 - 64; 128 + 24 + 64]
    // less than 256 in abs value, so takes 9 bits with sign
    // and new poly is in (2^63, 2^65)
    *scale = exp;
    return;
}

static vfloat INLINE __vexp_poly(vfloat z)
{
    vfloat zz =                         vSETf(EXP_C6);
           zz = vfma_vf_vf_vf_vf(zz, z, vSETf(EXP_C5));
           zz = vfma_vf_vf_vf_vf(zz, z, vSETf(EXP_C4));
           zz = vfma_vf_vf_vf_vf(zz, z, vSETf(EXP_C3));
           zz = vfma_vf_vf_vf_vf(zz, z, vSETf(EXP_C2));
           zz = vfma_vf_vf_vf_vf(zz, z, vSETf(EXP_C1));
           zz = vfma_vf_vf_vf_vf(zz, z, vSETf(EXP_C0));
    return zz;
}

static void INLINE
__vexp_kernel(vfloat vx, vfloat * vpoly, vfloat * vscale)
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
    vopmask mover   = vgt_vo_vf_vf(vx, vSETf(3.0f*EXP_HI));
    vx = vsel_vf_vo_vf_vf(mover, vSETf(3.0f*EXP_HI), vx);
    // exp underflows for x <= EXP_LO, it cannot be recovered with later
    // multiply by sincos, which is <= 1
    vopmask munder  = vle_vo_vf_vf(vx, vSETf(-3.0*EXP_HI));
    vx = vsel_vf_vo_vf_vf(munder, vSETf(-3.0*EXP_HI), vx);
    // tt = [x / ln2] = [x * log2(e)], convert to integer using right-shifter
    const vfloat vRS = vSETf(0x1.8p23);
    // least significant bits of t now contain an integer rounded
    // according to current rounding mode, default: to nearest
    // TODO: this algorithm will fail in directed rounding mode because
    // of the over/underestimate in t and thus tt.
    vfloat t = vfma_vf_vf_vf_vf(vx, vSETf(L2E), vRS);                           PRINT(t);
    // subtract right-shifter to obtain the integer as a normalized FP number
    vfloat tt= vsub_vf_vf_vf(t, vRS);

    // FMA is essential here. If no FMA, need to provide exact multiplication by
    // LN2_0, and this constant shall be changed to have e.g. 10 trailing zeros
    // so that the product can absorb the 10 bits of tt.
    vfloat z = vfma_vf_vf_vf_vf(tt, vSETf(-LN2_0), vx);
           z = vfma_vf_vf_vf_vf(tt, vSETf(-LN2_1), z);

    vint2 exponent = vF2I(t);
          // sign-extend integer exp:
          // wipe dummy FP sign, FP exponent field and two leading FP mantissa
          // bits (1 bit implicit), which are leftovers from right-shifter.
          exponent = vsll_vi2_vi2_i(exponent, 10);                              PRINT(exponent);
          exponent = vsra_vi2_vi2_i(exponent, 10);                              PRINT(exponent);

    // compute polynomial approximation, it shall be on the order
    // of exp() in the reduced range [-ln2/2, ln2/2], so zz is
    // in [1/sqrt(2), sqrt(2)] or in (1/2, 2)
    vfloat zz = __vexp_poly(z);                                                 PRINT(zz);

    // exponent scaling factor is now somewhere in [-128 - 24; 128 + 128 + 24].
    // Subtract 64 from scaling and add it back to polynomial so that later
    // polynomial * sin() always results in normalized numbers.
    // Plus it also makes exponent range symmetric:
    // [-128 - 24 - 64; 128 + 24 + 64], only 9 bits of
    // storage together with the sign.
    exponent = vsub_vi2_vi2_vi2(exponent, vSETi(64));                           PRINT(exponent);
    // new poly is in (2^63, 2^65)
    zz = vmul_vf_vf_vf(zz, vI2F(vSETi((64+127)<<23)));                          PRINT(zz);

    *vpoly = zz;
    *vscale = vI2F(exponent);
    return;
}
