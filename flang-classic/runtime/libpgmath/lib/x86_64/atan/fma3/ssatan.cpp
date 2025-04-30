
/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <immintrin.h>
#include "atan_defs.h"

extern "C" float __fss_atan_fma3(float);

float __fss_atan_fma3(float const sa) {
    __m128 a = _mm_set1_ps(sa);
/* P = fpminimax(atan(x),[|1,3,5,7,9,11,13,15,17|],[|double...|],[0.000000001;1.0]); */
    __m128 const VEC_INF = (__m128)_mm_set1_epi32(CONST_INF);
    __m128 const VEC_SGN = (__m128)_mm_set1_epi32(CONST_SGN);
    __m128 f_abs = _mm_and_ps(a, VEC_SGN);
    __m128 f_sgn = _mm_xor_ps(f_abs, a);
    __m128 inf_mask = _mm_cmp_ps(f_abs, VEC_INF, _CMP_EQ_OS);
    __m128 const PI_HALF = _mm_set1_ps(CONST_PIOVER2);
    __m256d const PI_HALF_D = _mm256_set1_pd(CONST_PIOVER2);

    __m128 x;


    __m128 f_rcp = _mm_rcp_ps(f_abs);

    __m256d d_abs = _mm256_cvtps_pd(f_abs);

    __m256d d_rcp = _mm256_cvtps_pd(f_rcp);

    __m256d const VECD_CUT = _mm256_set1_pd(1.0);

    __m256d d_x = _mm256_fnmadd_pd(d_rcp, d_abs, VECD_CUT);
            d_x= _mm256_fmadd_pd(d_x,d_x,d_x);
    __m256d rro_mask = _mm256_cmp_pd(d_abs, VECD_CUT, _CMP_GT_OS);
            d_x = _mm256_fmadd_pd(d_rcp,d_x,d_rcp);
            d_x = _mm256_blendv_pd(d_abs, d_x, rro_mask);

    __m256d const C0 = _mm256_set1_pd(DBL17_C0);
    __m256d const C1 = _mm256_set1_pd(DBL17_C1);
    __m256d const C2 = _mm256_set1_pd(DBL17_C2);
    __m256d const C3 = _mm256_set1_pd(DBL17_C3);
    __m256d const C4 = _mm256_set1_pd(DBL17_C4);
    __m256d const C5 = _mm256_set1_pd(DBL17_C5);
    __m256d const C6 = _mm256_set1_pd(DBL17_C6);
    __m256d const C7 = _mm256_set1_pd(DBL17_C7);
    __m256d const C8 = _mm256_set1_pd(DBL17_C8);

    __m256d x2 = _mm256_mul_pd(d_x, d_x);

    __m256d A3 = _mm256_fmadd_pd(x2, C8, C7);
    __m256d A2 = _mm256_fmadd_pd(x2, C5, C4);
    __m256d A1 = _mm256_fmadd_pd(x2, C2, C1);

    __m256d x6 = _mm256_mul_pd(x2, x2);

            A3 = _mm256_fmadd_pd(x2, A3, C6);
            A2 = _mm256_fmadd_pd(x2, A2, C3);
            A1 = _mm256_fmadd_pd(x2, A1, C0);

            x6 = _mm256_mul_pd(x6, x2);

            A2 = _mm256_fmadd_pd(A3, x6, A2);

            A1 = _mm256_fmadd_pd(A2, x6, A1);

            d_x = _mm256_mul_pd(d_x, A1);

    __m256d t = _mm256_sub_pd(PI_HALF_D, d_x);
    d_x = _mm256_blendv_pd(d_x, t, rro_mask);
    x = _mm256_cvtpd_ps(d_x);

    x = _mm_blendv_ps(x, PI_HALF, inf_mask);

    x = _mm_or_ps(x, f_sgn);
        return _mm_cvtss_f32(x);
}

