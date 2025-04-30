
/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <immintrin.h>
#include "atan_defs.h"

#define LO 0
#define HI 1

extern "C" __m256 __fvs_atan_fma3_256(__m256 const a);

inline __m256 dtf(__m256d dh, __m256d dl) {
    __m128 xlo = _mm256_cvtpd_ps(dl);
    __m128 xhi = _mm256_cvtpd_ps(dh);
    __m256 x = _mm256_castps128_ps256(xlo);
           x = _mm256_insertf128_ps(x, xhi,HI);
    return x;
}


__m256 __fvs_atan_fma3_256(__m256 const a) {
/* P = fpminimax(atan(x),[|1,3,5,7,9,11,13,15,17|],[|double...|],[0.000000001;1.0]); */
    __m256 const VEC_INF = (__m256)_mm256_set1_epi32(CONST_INF);
    __m256 const VEC_SGN = (__m256)_mm256_set1_epi32(CONST_SGN);
    __m256 abs = _mm256_and_ps(a, VEC_SGN);
    __m256 sgn = _mm256_xor_ps(abs, a);
    __m256 inf_mask = _mm256_cmp_ps(abs, VEC_INF, _CMP_EQ_OS);
    __m256 const PI_HALF = _mm256_set1_ps(CONST_PIOVER2);
    __m256d const PI_HALF_D = _mm256_set1_pd(CONST_PIOVER2);

    __m256 x;

    __m128 f_abs_hi = _mm256_extractf128_ps(abs,1);
    __m128 f_abs_lo = _mm256_extractf128_ps(abs,0);

    __m128 f_rcp_hi = _mm_rcp_ps(f_abs_hi);
    __m128 f_rcp_lo = _mm_rcp_ps(f_abs_lo);

    __m256d abs_hi = _mm256_cvtps_pd(f_abs_hi);
    __m256d abs_lo = _mm256_cvtps_pd(f_abs_lo);

    __m256d rcp_hi = _mm256_cvtps_pd(f_rcp_hi);
    __m256d rcp_lo = _mm256_cvtps_pd(f_rcp_lo);

    __m256d const VECD_CUT = _mm256_set1_pd(CONST_ONE);

    __m256d xd_hi = _mm256_fnmadd_pd(rcp_hi, abs_hi, VECD_CUT);
    __m256d xd_lo = _mm256_fnmadd_pd(rcp_lo, abs_lo, VECD_CUT);
    __m256d rro_mask_lo = _mm256_cmp_pd(abs_lo, VECD_CUT, _CMP_GT_OS);
            xd_hi = _mm256_fmadd_pd(xd_hi, xd_hi, xd_hi);
            xd_lo = _mm256_fmadd_pd(xd_lo, xd_lo, xd_lo);
    __m256d rro_mask_hi = _mm256_cmp_pd(abs_hi, VECD_CUT, _CMP_GT_OS);
            xd_hi = _mm256_fmadd_pd(rcp_hi,xd_hi,rcp_hi);
            xd_lo = _mm256_fmadd_pd(rcp_lo,xd_lo,rcp_lo);

            xd_hi = _mm256_blendv_pd(abs_hi, xd_hi, rro_mask_hi);
            xd_lo = _mm256_blendv_pd(abs_lo, xd_lo, rro_mask_lo);

    __m256d const C0 = _mm256_set1_pd(DBL17_C0);
    __m256d const C1 = _mm256_set1_pd(DBL17_C1);
    __m256d const C2 = _mm256_set1_pd(DBL17_C2);
    __m256d const C3 = _mm256_set1_pd(DBL17_C3);
    __m256d const C4 = _mm256_set1_pd(DBL17_C4);
    __m256d const C5 = _mm256_set1_pd(DBL17_C5);
    __m256d const C6 = _mm256_set1_pd(DBL17_C6);
    __m256d const C7 = _mm256_set1_pd(DBL17_C7);
    __m256d const C8 = _mm256_set1_pd(DBL17_C8);
    
    __m256d x2_hi = _mm256_mul_pd(xd_hi, xd_hi);
    __m256d x2_lo = _mm256_mul_pd(xd_lo, xd_lo);

    __m256d A3_hi = _mm256_fmadd_pd(x2_hi, C8, C7);
    __m256d A2_hi = _mm256_fmadd_pd(x2_hi, C5, C4);
    __m256d A1_hi = _mm256_fmadd_pd(x2_hi, C2, C1);
    __m256d A3_lo = _mm256_fmadd_pd(x2_lo, C8, C7);
    __m256d A2_lo = _mm256_fmadd_pd(x2_lo, C5, C4);
    __m256d A1_lo = _mm256_fmadd_pd(x2_lo, C2, C1);

    __m256d x6_hi = _mm256_mul_pd(x2_hi, x2_hi);
    __m256d x6_lo = _mm256_mul_pd(x2_lo, x2_lo);

            A3_hi = _mm256_fmadd_pd(x2_hi, A3_hi, C6);
            A2_hi = _mm256_fmadd_pd(x2_hi, A2_hi, C3);
            A1_hi = _mm256_fmadd_pd(x2_hi, A1_hi, C0);
            A3_lo = _mm256_fmadd_pd(x2_lo, A3_lo, C6);
            A2_lo = _mm256_fmadd_pd(x2_lo, A2_lo, C3);
            A1_lo = _mm256_fmadd_pd(x2_lo, A1_lo, C0);

            x6_hi = _mm256_mul_pd(x6_hi, x2_hi);
            x6_lo = _mm256_mul_pd(x6_lo, x2_lo);

            A2_hi = _mm256_fmadd_pd(A3_hi, x6_hi, A2_hi);
            A2_lo = _mm256_fmadd_pd(A3_lo, x6_lo, A2_lo);

            A1_hi = _mm256_fmadd_pd(A2_hi, x6_hi, A1_hi);
            A1_lo = _mm256_fmadd_pd(A2_lo, x6_lo, A1_lo);

            xd_hi = _mm256_mul_pd(xd_hi, A1_hi);
            xd_lo = _mm256_mul_pd(xd_lo, A1_lo);

    __m256d t_hi = _mm256_sub_pd(PI_HALF_D, xd_hi);
    __m256d t_lo = _mm256_sub_pd(PI_HALF_D, xd_lo);
    xd_hi = _mm256_blendv_pd(xd_hi, t_hi, rro_mask_hi);
    xd_lo = _mm256_blendv_pd(xd_lo, t_lo, rro_mask_lo);
    x = dtf(xd_hi, xd_lo);

    x = _mm256_blendv_ps(x, PI_HALF, inf_mask);

    x = _mm256_or_ps(x, sgn);
    
    return x;
}

