
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "math_common.h"
#include "sleef_common.h"

static float INLINE __ldexpf_scalar_kernel(float a, int scale)
{
#if (defined __AVX512F__)
    float res = _mm_cvtss_f32(_mm_scalef_ss(_mm_set_ss(a), _mm_set_ss((float)scale)));
    return res;
#else
                                                                                PRINT(a); PRINT(scale);
    // Input is allowed to be such that signed |scale| < 256,
    // |a| may be in {+-0} or +-[2^-149, 2^0] as it comes from sin/cos,
    // but we took precaution outside this routine and normalized a,
    // so that it is within +-[2^-149 + 64, 2^64] or zero.

    // Zeros and Inf/NaNs are handled separately.
    // Input denormals end up here too and yield incorrect result.
    // FIXME: assert(this function assumes no denormals on input !!!);
    unsigned exp_bits = F2I(a) & FL_EXP_MASK;
    unsigned zeroinfnan_mask = ((exp_bits == FL_EXP_MASK) || (exp_bits == 0))
                             ? 0xffffffff : 0;                                  PRINT(zeroinfnan_mask);
    // Preserve sign of input, quiet NaN
    float zeroinfnan_res = a + a;                                               PRINT(zeroinfnan_res);

    // biased exponent bits, shifted to least significant position
    unsigned getexp_a = exp_bits >> (FL_PREC_BITS-1);                           PRINT(getexp_a);

    // For a * 2^scale to fit in floats we need getexp(a) + scale
    // to fit in exponents range of floats: bias + (FL_EXP_MIN-1, FL_EXP_MAX).
    // FL_EXP_MIN-1 is less than the smallest denormal, but it may round up.
    int sumexp = getexp_a + scale;                                              PRINT(sumexp);

    // Return Inf of correct sign if overflow
    unsigned ovf_mask = ((sumexp > (signed int)(FL_EXP_MAX + FL_EXP_BIAS))
                      ? 0xffffffff : 0);
    unsigned sign_a   = F2I(a) & FL_SIGN_BIT;                                   PRINT(sign_a);
    unsigned ovf_res  = (sign_a | FL_EXP_MASK);                                 PRINT(ovf_res);

    // If underflow, return zero of correct sign
    unsigned udf_mask = (sumexp < (signed int)(FL_EXP_MIN-1 + FL_EXP_BIAS) )
                      ? 0xffffffff : 0;
    unsigned udf_res  = sign_a;                                                 PRINT(udf_res);

    // Check if result is within denormalized numbers range
    // and doesn't completely underflow
    unsigned den_mask = ~udf_mask &
                      (((signed int)(sumexp) <= 0) ? 0xffffffff : 0);

    // If scaling leads to denormals: we shall do it via FP multiplication
    // 2^scale * a. But 2^scale alone may not be representable in FP, while
    // the product is OK. Thus we would like the sum of exponents sumexp in
    // range for FP. Since sumexp already contains the value of biased exponent
    // of a, we will first compensate a by reducing its exponent to biased zero:
    // a = a * 2^(-(getexp_a - bias)), or set exponent bits of a to FL_EXP_BIAS.
    // Now we would like sumexp become positive, for that we may add as little
    // as -(FL_EXP_MIN-2 + FL_EXP_BIAS). We'd have to compensate exponent of a
    // by this same quantity, so in the end we'll be setting exponent of a to
    // FL_EXP_BIAS + (FL_EXP_MIN-2 + FL_EXP_BIAS) = 2*FL_EXP_BIAS + FL_EXP_MIN-2
    int new_scale = ((unsigned int)(sumexp -(FL_EXP_MIN-2 + FL_EXP_BIAS)))
                                                          << (FL_PREC_BITS-1);  PRINT(new_scale);
    float new_a = I2F((F2I(a) & (~FL_EXP_MASK)) |
                       ((2*FL_EXP_BIAS + FL_EXP_MIN-2) << (FL_PREC_BITS-1)));   PRINT(new_a);
    float den_res = new_a * I2F(new_scale);                                     PRINT(den_res);

    // normal case, just add scale to exponent bits
    unsigned gen_res = F2I(a) + (((unsigned int)scale) << (FL_PREC_BITS-1));    PRINT(gen_res);
    unsigned gen_mask = ~(ovf_mask | udf_mask | den_mask);

    float result = I2F((F2I(zeroinfnan_res) & zeroinfnan_mask) |
          ((~zeroinfnan_mask) & ((ovf_res & ovf_mask) |
                                 (udf_res & udf_mask) |
                                 (F2I(den_res) & den_mask) |
                                 (gen_res & gen_mask))));                       PRINT(result);

    return result;
#endif //#if (defined __AVX512F__)
}

static vfloat INLINE 
//static vfloat __attribute__((noinline))
__vldexpf_manual(vfloat va, vfloat vscale)
{
                                                                                PRINT(va); PRINT(vscale);
    // Input is allowed to be such that signed |scale| < 256,
    // |a| may be in {+-0} or +-[2^-149, 2^0] as it comes from sin/cos,
    // but we took precaution outside this routine and normalized a,
    // so that it is within +-[2^-149 + 64, 2^64] or zero.

    // Zeros and Inf/NaNs are handled separately.
    // Input denormals end up here too and yield incorrect result.
    // FIXME: assert(this function assumes no denormals on input !!!);
    vint2 exp_bits = vand_vi2_vi2_vi2(vF2I(va), vSETi(FL_EXP_MASK));
    vopmask zero_mask = veq_vo_vi2_vi2(exp_bits, vSETi(0));
    vopmask infnan_mask = veq_vo_vi2_vi2(exp_bits, vSETi(FL_EXP_MASK));
    vopmask zeroinfnan_mask = vor_vo_vo_vo(zero_mask, infnan_mask);             PRINT(zeroinfnan_mask);

    // Preserve sign of input, quiet NaN
    vfloat zeroinfnan_res = vadd_vf_vf_vf(va, va);                              PRINT(zeroinfnan_res);

    // biased exponent bits, shifted to least significant position
    vint2 getexp_a = vsrl_vi2_vi2_i(exp_bits, FL_PREC_BITS-1);                  PRINT(getexp_a);

    // For a * 2^scale to fit in floats we need getexp(a) + scale
    // to fit in exponents range of floats: bias + (FL_EXP_MIN-1, FL_EXP_MAX).
    // FL_EXP_MIN-1 is less than the smallest denormal, but it may round up.
    vint2 sumexp = vadd_vi2_vi2_vi2(getexp_a, vF2I(vscale));                    PRINT(sumexp);

    // Return Inf of correct sign if overflow
    vopmask ovf_mask = vgt_vo_vi2_vi2(sumexp, vSETi(FL_EXP_MAX + FL_EXP_BIAS));
    vint2 sign_a = vand_vi2_vi2_vi2(vF2I(va), vSETi(FL_SIGN_BIT));              PRINT(sign_a);
    vint2 ovf_res = vor_vi2_vi2_vi2(sign_a, vSETi(FL_EXP_MASK));                PRINT(ovf_res);

    // If underflow, return zero of correct sign
    vopmask udf_mask = vgt_vo_vi2_vi2(vSETi(FL_EXP_MIN-1 + FL_EXP_BIAS), sumexp);
    vint2 udf_res = sign_a;                                                     PRINT(udf_res);

    // Check if result is within denormalized numbers range
    // and doesn't completely underflow
    vopmask den_mask = vandnot_vo_vo_vo(udf_mask, vgt_vo_vi2_vi2(vSETi(1), sumexp));

    // If scaling leads to denormals: we shall do it via FP multiplication
    // 2^scale * a. But 2^scale alone may not be representable in FP, while
    // the product is OK. Thus we would like the sum of exponents sumexp in
    // range for FP. Since sumexp already contains the value of biased exponent
    // of a, we will first compensate a by reducing its exponent to biased zero:
    // a = a * 2^(-(getexp_a - bias)), or set exponent bits of a to FL_EXP_BIAS.
    // Now we would like sumexp become positive, for that we may add as little
    // as -(FL_EXP_MIN-2 + FL_EXP_BIAS). We'd have to compensate exponent of a
    // by this same quantity, so in the end we'll be setting exponent of a to
    // FL_EXP_BIAS + (FL_EXP_MIN-2 + FL_EXP_BIAS) = 2*FL_EXP_BIAS + FL_EXP_MIN-2
    vint2 new_scale =
            vsll_vi2_vi2_i(
              vadd_vi2_vi2_vi2(sumexp, vSETi(-(FL_EXP_MIN-2 + FL_EXP_BIAS))),
              FL_PREC_BITS-1);                                                  PRINT(new_scale);
    vfloat new_a = vI2F(vor_vi2_vi2_vi2(
                   vand_vi2_vi2_vi2(vF2I(va), vSETi(~FL_EXP_MASK)),
                   vSETi((2*FL_EXP_BIAS + FL_EXP_MIN-2) << (FL_PREC_BITS-1)))); PRINT(new_a);
    vfloat den_res = vmul_vf_vf_vf(new_a, vI2F(new_scale));                     PRINT(den_res);

    // normal case, just add scale to exponent bits
    vint2 gen_res = vadd_vi2_vi2_vi2(vF2I(va),
                        vsll_vi2_vi2_i( vF2I(vscale), FL_PREC_BITS-1));         PRINT(gen_res);
    vopmask ngen_mask =
                  vor_vo_vo_vo(vor_vo_vo_vo(ovf_mask, udf_mask), den_mask);

    vfloat result = vI2F(
           vor_vi2_vi2_vi2(
             vand_vi2_vo_vi2(zeroinfnan_mask, vF2I(zeroinfnan_res)),
             vandnot_vi2_vo_vi2(zeroinfnan_mask,
                vor_vi2_vi2_vi2(
                  vand_vi2_vo_vi2(ovf_mask, ovf_res),
                  vor_vi2_vi2_vi2(
                    vand_vi2_vo_vi2(udf_mask, udf_res),
                    vor_vi2_vi2_vi2(
                      vand_vi2_vo_vi2(den_mask, vF2I(den_res)),
                      vandnot_vi2_vo_vi2(ngen_mask, gen_res)))))));             PRINT(result);

    return result;
}

static vfloat INLINE
//static vfloat __attribute__((noinline))
__vldexpf_kernel(vfloat va, vfloat vscale)
{
    PRINT(va); PRINT(vscale);
#if (defined __AVX512F__) && ((defined __AVX512VL__) || (_VL == 8))
    // use AVX512VL instruction for _VL < 8
    // use AVX512F instruction in case of a full width
    vfloat vfres = JOIN(__SIMD_TYPE,_scalef_ps)(va, vcast_vf_vi2(vF2I(vscale)));            PRINT(vfres);
    return vfres;
#elif (defined __AVX512F__)
    // AVX512VL not supported and _VL < 8
    vfloat vscale_converted = vcast_vf_vi2(vF2I(vscale));                                   PRINT(vscale_converted);
    __mmask16 mask = (__mmask16)((1 << (2*_VL)) - 1);                                       PRINT(mask);
    __m512 fullwidth_va     = JOIN3(_mm512_castps,__SIMD_BITS,_ps512)(va);                  PRINT(fullwidth_va);
    __m512 fullwidth_vscale = JOIN3(_mm512_castps,__SIMD_BITS,_ps512)(vscale_converted);    PRINT(fullwidth_vscale);
    __m512 fullwidth_vfres  = _mm512_maskz_scalef_ps(mask, fullwidth_va, fullwidth_vscale); PRINT(fullwidth_vfres);
    vfloat vfres = JOIN(_mm512_castps512_ps,__SIMD_BITS)(fullwidth_vfres);                  PRINT(vfres);
    return vfres;
#else
    return __vldexpf_manual(va, vscale);
#endif
}
