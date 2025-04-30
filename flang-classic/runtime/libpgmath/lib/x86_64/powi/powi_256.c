/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file powi_avx2.c
 * AVX2 - 256 bit implementation of R(:)**I(:).
 */


#include <immintrin.h>
#include "mth_intrinsics.h"

/**
 *  \brief Compute R4(:)**I4(:)
 *  \param[in] _vx (__m256)  R4(:)
 *  \param[in] _vi (__m256i) I4(:)
 *  \return (__m256) R4(:)**I4(:)
 */

vrs8_t
__fs_powi_8_avx2(vrs8_t _vx, vis8_t _vi)
{
    __m256  vx = (__m256)_vx;
    __m256i vi = (__m256i)_vi;
    __m256  vt;
    __m256  vr;
    __m256i vm;
    __m256i vj;
    __m256  vf1p0 = _mm256_set1_ps(1.0);
    __m256i vi0   = _mm256_setzero_si256();
    __m256i vlsb  = _mm256_set1_epi32(1);

    vj = _mm256_abs_epi32(vi);
    vr = vf1p0;

    vm = _mm256_cmpgt_epi32(vj, vi0);
    if (0 == _mm256_movemask_epi8(vm)) return vr;

    for (;;) {
        vm = _mm256_and_si256(vj, vlsb);
        vm = _mm256_sub_epi32(vi0, vm);
        // Where vm == -1, vt = vx, else 1.0
        vt = _mm256_blendv_ps(vf1p0, vx, (__m256)vm);
        vr = _mm256_mul_ps(vr, vt);
        vj = _mm256_srli_epi32(vj, 1);
        vm = _mm256_cmpgt_epi32(vj, vi0);
        if (0 == _mm256_movemask_epi8(vm)) break;
        vx = _mm256_blendv_ps(vf1p0, vx, (__m256)vm);
        vx = _mm256_mul_ps(vx, vx);
    }

    if (0 != _mm256_movemask_ps((__m256)vi)) {
        vx = _mm256_div_ps(vf1p0, vr);
        vr = _mm256_blendv_ps(vr, vx, (__m256)vi);
    }

    return vr;
}

/**
 *  \brief Compute R4(:)**I4(:) under mask
 *  \param[in] _vx (__m256)  R4(:)
 *  \param[in] _vi (__m256i) I4(:)
 *  \param[in] _vm (__m256i) I4(:)
 *  \return (__m256) WHERE(_vm(:) != 0) R4(:)**I4(:)
 */
vrs8_t
__fs_powi_8m_avx2(vrs8_t _vx, vis8_t _vi, vis8_t _vm)
{
    __m256  vx = (__m256)_vx;
    /*
     * Intentionally use PS for integer values, simplifies numer of arguments
     * that need to be cast when using _mm256_blendv_ps() intrinsic.
     */
    __m256  vi = (__m256)_vi;
    __m256  vm = (__m256)_vm;

    vx = _mm256_blendv_ps(_mm256_set1_ps(0.0), vx, vm);
    vi = _mm256_blendv_ps((__m256)_mm256_set1_epi32(0), vi, vm);

    return __fs_powi_8_avx2((vrs8_t)vx, (vis8_t)vi);
}

/**
 *  \brief Compute R8(:)**I8(:)
 *  \param[in] _vx (__m256d) R8(:)
 *  \param[in] _vi (__m256i) I8(:)
 *  \return (__m256d) R4(:)**I8(:)
 */
vrd4_t
__fd_powk_4_avx2(vrd4_t _vx, vid4_t _vi)
{
    __m256d vx = (__m256d)_vx;
    __m256i vi = (__m256i)_vi;
    __m256d vt;
    __m256d vr;
    __m256i vm;
    __m256i vj;
    __m256d vf1p0 = _mm256_set1_pd(1.0);
    __m256i vi0   = _mm256_setzero_si256();
    __m256i vlsb  = _mm256_set1_epi64x(1);

    vj = _mm256_sub_epi64(vi0, vi);
    vm = _mm256_cmpgt_epi64(vi0, vi);
    vj = (__m256i)_mm256_blendv_pd((__m256d)vi, (__m256d)vj, (__m256d)vm);
    vr = vf1p0;

    vm = _mm256_cmpgt_epi64(vj, vi0);
    if (0 == _mm256_movemask_epi8(vm)) return vr;

    for (;;) {
        vm = _mm256_and_si256(vj, vlsb);
        vm = _mm256_sub_epi64(vi0, vm);
        // Where vm == -1, vt = vx, else 1.0
        vt = _mm256_blendv_pd(vf1p0, vx, (__m256d)vm);
        vr = _mm256_mul_pd(vr, vt);
        vj = _mm256_srli_epi64(vj, 1);
        vm = _mm256_cmpgt_epi64(vj, vi0);
        if (0 == _mm256_movemask_epi8(vm)) break;
        vx = _mm256_blendv_pd(vf1p0, vx, (__m256d)vm);
        vx = _mm256_mul_pd(vx, vx);
    }

    if (0 != _mm256_movemask_pd((__m256d)vi)) {
        vx = _mm256_div_pd(vf1p0, vr);
        vr = _mm256_blendv_pd(vr, vx, (__m256d)vi);
    }

    return vr;
}

/**
 *  \brief Compute R8(:)**I8(:) under mask
 *  \param[in] _vx (__m256d) R8(:)
 *  \param[in] _vi (__m256i) I8(:)
 *  \param[in] _vm (__m256i) I8(:)
 *  \return (__m256d) WHERE(_vm(:) != 0) R8(:)**I8(:)
 */
vrd4_t
__fd_powk_4m_avx2(vrd4_t _vx, vid4_t _vi, vid4_t _vm)
{
    __m256d vx = (__m256d)_vx;
    /*
     * Intentionally use PD for integer values, simplifies numer of arguments
     * that need to be cast when using _mm256_blendv_pd() intrinsic.
     */
    __m256d vi = (__m256d)_vi;
    __m256d vm = (__m256d)_vm;

    vx = _mm256_blendv_pd(_mm256_set1_pd(0.0), vx, vm);
    vi = _mm256_blendv_pd((__m256d)_mm256_set1_epi64x(0), vi, vm);

    return __fd_powk_4_avx2((vrd4_t)vx, (vid4_t)vi);
}

/**
 *  \brief Compute R8(:)**I4(:)
 *  \param[in] _vx (__m256d) R8(:)
 *  \param[in] _vi (__m256i) I4(:)
 *  \return (__m256d) R8(:)**I4(:)
 */
vrd4_t
__fd_powi_4_avx2(vrd4_t _vx, vis4_t _vi)
{
    return __fd_powk_4_avx2(_vx, (vid4_t)_mm256_cvtepi32_epi64((__m128i)_vi));
}

/**
 *  \brief Compute R8(:)**I8(:) under mask
 *  \param[in] _vx (__m256d) R8(:)
 *  \param[in] _vi (__m256i) I4(:)
 *  \param[in] _vm (__m256i) I8(:)
 *  \return (__m256d) WHERE(_vm(:) != 0) R8(:)**I8(:)
 */
vrd4_t
__fd_powi_4m_avx2(vrd4_t _vx, vis4_t _vi, vid4_t _vm)
{
    return __fd_powk_4m_avx2(_vx, (vid4_t)_mm256_cvtepi32_epi64((__m128i)_vi), _vm);
}

/**
 *  \brief (internal) Kernel to compute R4(0:3)**I8_lower(:), Compute R4(4:7)**I8_upper(:)
 *  \param[in] _vx (__m256d) R4(:)
 *  \param[in] _vl (__m256i) I8(:)
 *  \param[in] _vu (__m256i) I8(:)
 *  \return (__m256d) (R4(4:7)**I8_upper(:))<<128 | R4(0:3)**I8_lower(:)
 */
vrs8_t
__fs_powk_2x4_avx2 (vrs8_t _vx, vid4_t _vl, vid4_t _vu)
{
    __m256  vx;
    __m256  vt;
    __m256  vr;
    __m256  vrl;
    __m256  vf1p0 = _mm256_set1_ps(1.0);
    __m256i vmi;    // Inner loop mask
    __m256i vmo;    // Outer loop mask
    __m256i vi;
    __m256i vj;
    __m256i vi0 = _mm256_setzero_si256();
    __m256i vlsb  = _mm256_set1_epi64x(1);
    __m128  vxmm;

    int     i;

    vxmm = _mm256_extractf128_ps(_vx, 0);
    vx = _mm256_insertf128_ps(_vx, _mm_shuffle_ps(vxmm, vxmm ,0xe), 1);
    vx = _mm256_shuffle_ps(vx, vx, 0x50);
    vi = (__m256i)_vl;

    for(i = 0 ; i < 2 ; i++) {
        vr  = vf1p0;
        vj  = _mm256_sub_epi64(vi0, vi);
        vmo = _mm256_cmpgt_epi64(vi0, vi);
        vj  = (__m256i)_mm256_blendv_pd((__m256d)vi, (__m256d)vj, (__m256d)vmo);
        for (;;) {
            vmi = _mm256_and_si256(vj, vlsb);
            vmi = _mm256_sub_epi64(vi0, vmi);
            vt  = (__m256) _mm256_blendv_pd((__m256d)vf1p0, (__m256d)vx, (__m256d)vmi);
            vr  = _mm256_mul_ps(vr, vt);
            vj  = _mm256_srli_epi64(vj, 1);
            vmi = _mm256_cmpgt_epi64(vj, vi0);
            if (0 == _mm256_movemask_epi8(vmi)) break;
            vx  = (__m256) _mm256_blendv_pd((__m256d)vf1p0, (__m256d)vx, (__m256d)vmi);
            vx  = _mm256_mul_ps(vx, vx);
        }
        if (0 != _mm256_movemask_pd((__m256d)vmo)) {
            vx = _mm256_div_ps(vf1p0, vr);
            vr = (__m256) _mm256_blendv_pd((__m256d)vr, (__m256d)vx, (__m256d)vmo);
        }
        if (i == 1) break;
        vrl = vr;
        vxmm = _mm256_extractf128_ps(_vx, 1);
        vx = _mm256_insertf128_ps(_vx, _mm_shuffle_ps(vxmm, vxmm ,0x40), 0);
        vx = _mm256_shuffle_ps(vx, vx, 0xfa);
        vi = (__m256i)_vu;
    }
    vrl =  _mm256_shuffle_ps(vrl, vrl, 0x8);
    vr  =  _mm256_shuffle_ps(vr, vr, 0x80);
    vt  = (__m256) _mm256_blend_pd((__m256d)vr, (__m256d)vrl, 0x5);
    return (vrs8_t) _mm256_permute4x64_pd((__m256d)vt, 0xd8);
}

/**
 *  \brief (external) Compute R4(0:3)**I8_lower(:), Compute R4(4:7)**I8_upper(:)
 *  \param[in] _vx (__m256d) R4(:)
 *  \param[in] _vl (__m256i) I8(:)
 *  \param[in] _vu (__m256i) I8(:)
 *  \return (__m256d) (R4(4:7)**I8_upper(:))<<128 | R4(0:3)**I8_lower(:)
 */
vrs8_t
__fs_powk_8_avx2(vrs8_t _vx, vid4_t _vl, vid4_t _vu)
{
    __m256i vu = (__m256i) _vu;
    __m256i vl = (__m256i) _vl;
    __m256i vabsu;
    __m256i vabsl;
    __m256i vi0 = _mm256_setzero_si256();
    __m256i vi2to31m1 = _mm256_set1_epi64x((1ll<<31)-1);
    __m256i vmu;
    __m256i vml;
    __m256i vi;

    vabsl = _mm256_sub_epi64(vi0, vl);
    vabsu = _mm256_sub_epi64(vi0, vu);
    vml   = _mm256_cmpgt_epi64(vi0, vl);
    vmu   = _mm256_cmpgt_epi64(vi0, vu);
    vabsl = (__m256i)_mm256_blendv_pd((__m256d)vl, (__m256d)vabsl, (__m256d)vml);
    vabsu = (__m256i)_mm256_blendv_pd((__m256d)vu, (__m256d)vabsu, (__m256d)vmu);

    vml = _mm256_cmpgt_epi64(vabsl, vi2to31m1);
    vmu = _mm256_cmpgt_epi64(vabsu, vi2to31m1);

    if (0 != _mm256_movemask_epi8(vmu|vml)) return __fs_powk_2x4_avx2(_vx, _vl, _vu);

    vabsl = _mm256_shuffle_epi32(vl, 0x08);
    vabsu = _mm256_shuffle_epi32(vu, 0x80);

    vi = _mm256_permute4x64_epi64 (_mm256_blend_epi32(vabsu, vabsl, 0x33), 0xd8);
    return __fs_powi_8_avx2(_vx, (vis8_t) vi);

}

/**
 *  \brief (external) Compute R4(0:3)**I8_lower(:), Compute R4(4:7)**I8_upper(:)
 *  \param[in] _vx (__m256d) R4(:)
 *  \param[in] _vl (__m256i) I8(:)
 *  \param[in] _vu (__m256i) I8(:)
 *  \param[in] _vm (__m256i) I8(:)
 *  \return (__m256d) (R4(4:7)**I8_upper(:))<<128 | R4(0:3)**I8_lower(:)
 */
vrs8_t
__fs_powk_8m_avx2(vrs8_t _vx, vid4_t _vl, vid4_t _vu, vis8_t _vm)
{
    __m256  vx = (__m256) _vx;
    /*
     * Intentionally use PD for integer values, simplifies numer of arguments
     * that need to be cast when using _mm256_blendv_pd() intrinsic.
     */
    __m256d vu = (__m256d) _vu;
    __m256d vl = (__m256d) _vl;
    __m256d vm = (__m256d) _vm;
    __m256d vt;

    vx = _mm256_blendv_ps(_mm256_set1_ps(0.0), vx, (__m256)_vm);
    vt = (__m256d)_mm256_cvtepi32_epi64((__m128i)_mm256_extractf128_pd(vm, 0));
    vl = _mm256_blendv_pd(_mm256_set1_pd(0), vl, vt);
    vt = (__m256d)_mm256_cvtepi32_epi64((__m128i)_mm256_extractf128_pd(vm, 1));
    vu = _mm256_blendv_pd(_mm256_set1_pd(0), vu, vt);
    return __fs_powk_8_avx2((vrs8_t)vx, (vid4_t)vl, (vid4_t)vu);
}
