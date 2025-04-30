
/* 
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */


#include <immintrin.h>
#include "atan_defs.h"
#include "mth_avx512helper.h"

#define LO 0
#define HI 1

extern "C" __m512 FCN_AVX512(__fvs_atan_fma3)(__m512 const a);

inline __m512 dtf(__m512d dh, __m512d dl) {
    __m256 xlo = _mm512_cvtpd_ps(dl);
    __m256 xhi = _mm512_cvtpd_ps(dh);
    __m512 x = _mm512_castps256_ps512(xlo);
           x = _MM512_INSERTF256_PS(x, xhi,HI);
    return x;
}


__m512 FCN_AVX512(__fvs_atan_fma3)(__m512 const a) {
/* P = fpminimax(atan(x),[|1,3,5,7,9,11,13,15,17|],[|double...|],[0.000000001;1.0]); */
    __m512 const VEC_INF = (__m512)_mm512_set1_epi32(CONST_INF);
    __m512 const VEC_SGN = (__m512)_mm512_set1_epi32(CONST_SGN);
    __m512 abs = _MM512_AND_PS(a, VEC_SGN);
    __m512 sgn = _MM512_XOR_PS(abs, a);
    __m512 inf_mask = _MM512_CMP_PS(abs, VEC_INF, _CMP_EQ_OS);
    __m512 const PI_HALF = _mm512_set1_ps(CONST_PIOVER2);
    __m512d const PI_HALF_D = _mm512_set1_pd(CONST_PIOVER2);

    __m512 x;

    __m256 f_abs_hi = _MM512_EXTRACTF256_PS(abs,1);
    __m256 f_abs_lo = _MM512_EXTRACTF256_PS(abs,0);

    __m256 f_rcp_hi = _mm256_rcp_ps(f_abs_hi);
    __m256 f_rcp_lo = _mm256_rcp_ps(f_abs_lo);

    __m512d abs_hi = _mm512_cvtps_pd(f_abs_hi);
    __m512d abs_lo = _mm512_cvtps_pd(f_abs_lo);

    __m512d rcp_hi = _mm512_cvtps_pd(f_rcp_hi);
    __m512d rcp_lo = _mm512_cvtps_pd(f_rcp_lo);

    __m512d const VECD_CUT = _mm512_set1_pd(CONST_ONE);

    __m512d xd_hi = _mm512_fnmadd_pd(rcp_hi, abs_hi, VECD_CUT);
    __m512d xd_lo = _mm512_fnmadd_pd(rcp_lo, abs_lo, VECD_CUT);
    __m512d rro_mask_lo = _MM512_CMP_PD(abs_lo, VECD_CUT, _CMP_GT_OS);
            xd_hi = _mm512_fmadd_pd(xd_hi, xd_hi, xd_hi);
            xd_lo = _mm512_fmadd_pd(xd_lo, xd_lo, xd_lo);
    __m512d rro_mask_hi = _MM512_CMP_PD(abs_hi, VECD_CUT, _CMP_GT_OS);
            xd_hi = _mm512_fmadd_pd(rcp_hi,xd_hi,rcp_hi);
            xd_lo = _mm512_fmadd_pd(rcp_lo,xd_lo,rcp_lo);

            xd_hi = _MM512_BLENDV_PD(abs_hi, xd_hi, rro_mask_hi);
            xd_lo = _MM512_BLENDV_PD(abs_lo, xd_lo, rro_mask_lo);

    __m512d const C0 = _mm512_set1_pd(DBL17_C0);
    __m512d const C1 = _mm512_set1_pd(DBL17_C1);
    __m512d const C2 = _mm512_set1_pd(DBL17_C2);
    __m512d const C3 = _mm512_set1_pd(DBL17_C3);
    __m512d const C4 = _mm512_set1_pd(DBL17_C4);
    __m512d const C5 = _mm512_set1_pd(DBL17_C5);
    __m512d const C6 = _mm512_set1_pd(DBL17_C6);
    __m512d const C7 = _mm512_set1_pd(DBL17_C7);
    __m512d const C8 = _mm512_set1_pd(DBL17_C8);
    
    __m512d x2_hi = _mm512_mul_pd(xd_hi, xd_hi);
    __m512d x2_lo = _mm512_mul_pd(xd_lo, xd_lo);

    __m512d A3_hi = _mm512_fmadd_pd(x2_hi, C8, C7);
    __m512d A2_hi = _mm512_fmadd_pd(x2_hi, C5, C4);
    __m512d A1_hi = _mm512_fmadd_pd(x2_hi, C2, C1);
    __m512d A3_lo = _mm512_fmadd_pd(x2_lo, C8, C7);
    __m512d A2_lo = _mm512_fmadd_pd(x2_lo, C5, C4);
    __m512d A1_lo = _mm512_fmadd_pd(x2_lo, C2, C1);

    __m512d x6_hi = _mm512_mul_pd(x2_hi, x2_hi);
    __m512d x6_lo = _mm512_mul_pd(x2_lo, x2_lo);

            A3_hi = _mm512_fmadd_pd(x2_hi, A3_hi, C6);
            A2_hi = _mm512_fmadd_pd(x2_hi, A2_hi, C3);
            A1_hi = _mm512_fmadd_pd(x2_hi, A1_hi, C0);
            A3_lo = _mm512_fmadd_pd(x2_lo, A3_lo, C6);
            A2_lo = _mm512_fmadd_pd(x2_lo, A2_lo, C3);
            A1_lo = _mm512_fmadd_pd(x2_lo, A1_lo, C0);

            x6_hi = _mm512_mul_pd(x6_hi, x2_hi);
            x6_lo = _mm512_mul_pd(x6_lo, x2_lo);

            A2_hi = _mm512_fmadd_pd(A3_hi, x6_hi, A2_hi);
            A2_lo = _mm512_fmadd_pd(A3_lo, x6_lo, A2_lo);

            A1_hi = _mm512_fmadd_pd(A2_hi, x6_hi, A1_hi);
            A1_lo = _mm512_fmadd_pd(A2_lo, x6_lo, A1_lo);

            xd_hi = _mm512_mul_pd(xd_hi, A1_hi);
            xd_lo = _mm512_mul_pd(xd_lo, A1_lo);

    __m512d t_hi = _mm512_sub_pd(PI_HALF_D, xd_hi);
    __m512d t_lo = _mm512_sub_pd(PI_HALF_D, xd_lo);
    xd_hi = _MM512_BLENDV_PD(xd_hi, t_hi, rro_mask_hi);
    xd_lo = _MM512_BLENDV_PD(xd_lo, t_lo, rro_mask_lo);
    x = dtf(xd_hi, xd_lo);

    x = _MM512_BLENDV_PS(x, PI_HALF, inf_mask);

    x = _MM512_OR_PS(x, sgn);
    
    return x;
}

