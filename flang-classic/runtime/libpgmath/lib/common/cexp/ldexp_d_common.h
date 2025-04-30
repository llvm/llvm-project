
/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "math_common.h"
#include "sleef_common.h"

static double INLINE __ldexp_d_scalar_kernel(double a, long long int scale)
{
#if (defined __AVX512F__)
    double res = _mm_cvtsd_f64(_mm_scalef_sd(_mm_set_sd(a), _mm_set_sd((double)scale)));
    return res;
#else
                                                                                PRINT(a); PRINT(scale);
    // Input is allowed to be such that signed |scale| < 2048,
    // |a| may be in {+-0} or +-[2^-1074, 2^0] as it comes from sin/cos,
    // but we took precaution outside this routine and normalized a,
    // so that it is within +-[2^-1074 + 512, 2^512] or zero.

    // Zeros and Inf/NaNs are handled separately.
    // Input denormals end up here too and yield incorrect result.
    // FIXME: assert(this function assumes no denormals on input !!!);
    unsigned long long int exp_bits = D2L(a) & DB_EXP_MASK;
    unsigned long long int zeroinfnan_mask =
        ((exp_bits == DB_EXP_MASK) || (exp_bits == 0ull))
                   ? -1ll : 0ll;                                                PRINT(zeroinfnan_mask);
    // Preserve sign of input, quiet NaN
    double zeroinfnan_res = a + a;                                              PRINT(zeroinfnan_res);

    // biased exponent bits, shifted to least significant position
    unsigned long long int getexp_a = exp_bits >> (DB_PREC_BITS-1);             PRINT(getexp_a);

    // For a * 2^scale to fit in double we need getexp(a) + scale
    // to fit in exponents range of double: bias + (DB_EXP_MIN-1, DB_EXP_MAX).
    // DB_EXP_MIN-1 is less than the smallest denormal, but it may round up.
    long long int sumexp = getexp_a + scale;                                    PRINT(sumexp);

    // Return Inf of correct sign if overflow
    unsigned long long int ovf_mask =
        (sumexp > (signed long long int)(DB_EXP_MAX + DB_EXP_BIAS))
                ? -1ll : 0ll;
    unsigned long long int sign_a   = D2L(a) & DB_SIGN_BIT;                     PRINT(sign_a);
    unsigned long long int ovf_res  = (sign_a | DB_EXP_MASK);                   PRINT(ovf_res);

    // If underflow, return zero of correct sign
    unsigned long long int udf_mask =
        (sumexp < (signed long long int)(DB_EXP_MIN-1 + DB_EXP_BIAS) )
                ? -1ll : 0ll;
    unsigned long long int udf_res  = sign_a;                                   PRINT(udf_res);

    // Check if result is within denormalized numbers range
    // and doesn't completely underflow
    unsigned long long int den_mask = ~udf_mask &
        (((signed long long int )(sumexp) <= 0ll) ? -1ll : 0ll);

    // If scaling leads to denormals: we shall do it via FP multiplication
    // 2^scale * a. But 2^scale alone may not be representable in FP, while
    // the product is OK. Thus we would like the sum of exponents sumexp in
    // range for FP. Since sumexp already contains the value of biased exponent
    // of a, we will first compensate a by reducing its exponent to biased zero:
    // a = a * 2^(-(getexp_a - bias)), or set exponent bits of a to DB_EXP_BIAS.
    // Now we would like sumexp become positive, for that we may add as little
    // as -(DB_EXP_MIN-2 + DB_EXP_BIAS). We'd have to compensate exponent of a
    // by this same quantity, so in the end we'll be setting exponent of a to
    // DB_EXP_BIAS + (DB_EXP_MIN-2 + DB_EXP_BIAS) = 2*DB_EXP_BIAS + DB_EXP_MIN-2
    long long int new_scale =
        ((unsigned long long int)(sumexp -(DB_EXP_MIN-2 + DB_EXP_BIAS)))
        << (DB_PREC_BITS-1);                                                    PRINT(new_scale);
    double new_a = L2D((D2L(a) & (~DB_EXP_MASK)) |
                     ((2*DB_EXP_BIAS + DB_EXP_MIN-2ll) << (DB_PREC_BITS-1)));   PRINT(new_a);
    double den_res = new_a * L2D(new_scale);                                    PRINT(den_res);

    // normal case, just add scale to exponent bits
    unsigned long long int gen_res = D2L(a) +
        (((unsigned long long int)scale) << (DB_PREC_BITS-1));                  PRINT(gen_res);
    unsigned long long int gen_mask = ~(ovf_mask | udf_mask | den_mask);

    double result = L2D((D2L(zeroinfnan_res) & zeroinfnan_mask) |
          ((~zeroinfnan_mask) & ((ovf_res & ovf_mask) |
                                 (udf_res & udf_mask) |
                                 (D2L(den_res) & den_mask) |
                                 (gen_res & gen_mask))));                       PRINT(result);

    return result;
#endif //#if (defined __AVX512F__)
}

static vdouble INLINE
//static vdouble __attribute__((noinline))
__vldexp_manual(vdouble va, vdouble vscale)
{
                                                                                PRINT(va); PRINT(vscale);
    // Input is allowed to be such that signed |scale| < 2048,
    // |a| may be in {+-0} or +-[2^-1074, 2^0] as it comes from sin/cos,
    // but we took precaution outside this routine and normalized a,
    // so that it is within +-[2^-1074 + 512, 2^512] or zero.

    // Zeros and Inf/NaNs are handled separately.
    // Input denormals end up here too and yield incorrect result.
    // FIXME: assert(this function assumes no denormals on input !!!);
    vint2 exp_bits = vand_vi2_vi2_vi2(vD2L(va), vSETll(DB_EXP_MASK));
    vopmask zero_mask = veq64_vo_vm_vm(exp_bits, vSETll(0));
    vopmask infnan_mask = veq64_vo_vm_vm(exp_bits, vSETll(DB_EXP_MASK));
    vopmask zeroinfnan_mask = vor_vo_vo_vo(zero_mask, infnan_mask);             PRINT(zeroinfnan_mask);

    // Preserve sign of input, quiet NaN
    vdouble zeroinfnan_res = vadd_vd_vd_vd(va, va);                             PRINT(zeroinfnan_res);

    // biased exponent bits, shifted to least significant position
    vint2 getexp_a = vsrl64_vi2_vi2_i(exp_bits, DB_PREC_BITS-1);                PRINT(getexp_a);

    // For a * 2^scale to fit in double we need getexp(a) + scale
    // to fit in exponents range of double: bias + (DB_EXP_MIN-1, DB_EXP_MAX).
    // DB_EXP_MIN-1 is less than the smallest denormal, but it may round up.
    vint2 sumexp = vadd64_vi2_vi2_vi2(getexp_a, vD2L(vscale));                  PRINT(sumexp);

    // Return Inf of correct sign if overflow
    vopmask ovf_mask = vgt64_vo_vm_vm(sumexp, vSETll(DB_EXP_MAX + DB_EXP_BIAS));
    vint2 sign_a = vand_vi2_vi2_vi2(vD2L(va), vSETll(DB_SIGN_BIT));             PRINT(sign_a);
    vint2 ovf_res = vor_vi2_vi2_vi2(sign_a, vSETll(DB_EXP_MASK));               PRINT(ovf_res);

    // If underflow, return zero of correct sign
    vopmask udf_mask = vgt64_vo_vm_vm(vSETll(DB_EXP_MIN-1 + DB_EXP_BIAS), sumexp);
    vint2 udf_res = sign_a;                                                     PRINT(udf_res);

    // Check if result is within denormalized numbers range
    // and doesn't completely underflow
    vopmask den_mask = vandnot_vo_vo_vo(udf_mask, vgt64_vo_vm_vm(vSETll(1), sumexp));

    // If scaling leads to denormals: we shall do it via FP multiplication
    // 2^scale * a. But 2^scale alone may not be representable in FP, while
    // the product is OK. Thus we would like the sum of exponents sumexp in
    // range for FP. Since sumexp already contains the value of biased exponent
    // of a, we will first compensate a by reducing its exponent to biased zero:
    // a = a * 2^(-(getexp_a - bias)), or set exponent bits of a to DB_EXP_BIAS.
    // Now we would like sumexp become positive, for that we may add as little
    // as -(DB_EXP_MIN-2 + DB_EXP_BIAS). We'd have to compensate exponent of a
    // by this same quantity, so in the end we'll be setting exponent of a to
    // DB_EXP_BIAS + (DB_EXP_MIN-2 + DB_EXP_BIAS) = 2*DB_EXP_BIAS + DB_EXP_MIN-2
    vint2 new_scale =
            vsll64_vi2_vi2_i(
              vadd_vi2_vi2_vi2(sumexp, vSETll(-(DB_EXP_MIN-2 + DB_EXP_BIAS))),
              DB_PREC_BITS-1);                                                  PRINT(new_scale);
    vdouble new_a = vL2D(vor_vi2_vi2_vi2(
                   vand_vi2_vi2_vi2(vD2L(va), vSETll(~DB_EXP_MASK)),
                   vSETll((2ULL*DB_EXP_BIAS + DB_EXP_MIN-2) << (DB_PREC_BITS-1)))); PRINT(new_a);
    vdouble den_res = vmul_vd_vd_vd(new_a, vL2D(new_scale));                    PRINT(den_res);

    // normal case, just add scale to exponent bits
    vint2 gen_res = vadd_vi2_vi2_vi2(vD2L(va),
                        vsll64_vi2_vi2_i( vD2L(vscale), DB_PREC_BITS-1));       PRINT(gen_res);
    vopmask ngen_mask =
                  vor_vo_vo_vo(vor_vo_vo_vo(ovf_mask, udf_mask), den_mask);

    vdouble result = vL2D(
           vor_vi2_vi2_vi2(
             vand_vm_vo64_vm(zeroinfnan_mask, vD2L(zeroinfnan_res)),
             vandnot_vm_vo64_vm(zeroinfnan_mask,
                vor_vi2_vi2_vi2(
                  vand_vm_vo64_vm(ovf_mask, ovf_res),
                  vor_vi2_vi2_vi2(
                    vand_vm_vo64_vm(udf_mask, udf_res),
                    vor_vi2_vi2_vi2(
                      vand_vm_vo64_vm(den_mask, vD2L(den_res)),
                      vandnot_vm_vo64_vm(ngen_mask, gen_res)))))));             PRINT(result);

    return result;
}

static vdouble INLINE
//static vdouble __attribute__((noinline))
__vldexp_kernel(vdouble va, vdouble vscale)
{
    PRINT(va); PRINT(vscale);
#if (defined __AVX512F__) && (defined __AVX512VL__) && (defined __AVX512DQ__)
    vdouble vfres = JOIN(__SIMD_TYPE,_scalef_pd)(va, JOIN(__SIMD_TYPE,_cvtepi64_pd)(vD2L(vscale)));           PRINT(vfres);
    return vfres;
#elif (defined __AVX512F__)
    __mmask8 mask = (__mmask8)((1 << (2*_VL)) - 1);                                          PRINT(mask);
    #define _mm512_castpd512_pd512(x) (x) // no cast operation needed if in full width
    __m512d fullwidth_va     = JOIN3(_mm512_castpd,__SIMD_BITS,_pd512)(va);                  PRINT(fullwidth_va);
    __m512d fullwidth_vscale = JOIN3(_mm512_castpd,__SIMD_BITS,_pd512)(vscale);              PRINT(fullwidth_vscale);
    // need to emulate conversion from signed 64-bit integer to double
    // we know that |scale| < 2^31, so the trick works
    __m512d fullwidth_vscale_dp =
            _mm512_castsi512_pd(
                _mm512_add_epi32(
                    _mm512_castpd_si512(fullwidth_vscale),
                    _mm512_castpd_si512(_mm512_set1_pd(0x1.8p52))
                )
            );                                                                               PRINT(fullwidth_vscale_dp);
    fullwidth_vscale_dp =
            _mm512_maskz_sub_pd(mask, fullwidth_vscale_dp, _mm512_set1_pd(0x1.8p52));        PRINT(fullwidth_vscale_dp);
    __m512d fullwidth_vfres =
            _mm512_maskz_scalef_pd(mask, fullwidth_va, fullwidth_vscale_dp);                 PRINT(fullwidth_vfres);
    vdouble vfres = JOIN(_mm512_castpd512_pd,__SIMD_BITS)(fullwidth_vfres);                  PRINT(vfres);
    #undef _mm512_castpd512_pd512
    return vfres;
#else
    return __vldexp_manual(va, vscale);
#endif
}
