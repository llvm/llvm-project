/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
/** \file powi_sse4.c
 * sse4 - 256 bit implementation of R(:)**I(:).
 */


#include <immintrin.h>
#include "mth_intrinsics.h"

/**
 *  \brief Compute R4(:)**I4(:)
 *  \param[in] _vx (__m128)  R4(:)
 *  \param[in] _vi (__m128i) I4(:)
 *  \return (__m128) R4(:)**I4(:)
 */

vrs4_t
__fs_powi_4_sse4(vrs4_t _vx, vis4_t _vi)
{
    __m128  vx = (__m128)_vx;
    __m128i vi = (__m128i)_vi;
    __m128  vt;
    __m128  vr;
    __m128i vm;
    __m128i vj;
    __m128  vf1p0 = _mm_set1_ps(1.0);
    __m128i vi0   = _mm_setzero_si128();
    __m128i vlsb  = _mm_set1_epi32(1);

    vj = _mm_abs_epi32(vi);
    vr = vf1p0;

    vm = _mm_cmpgt_epi32(vj, vi0);
    if (0 == _mm_movemask_epi8(vm)) return vr;

    for (;;) {
        vm = _mm_and_si128(vj, vlsb);
        vm = _mm_sub_epi32(vi0, vm);
        // Where vm == -1, vt = vx, else 1.0
        vt = _mm_blendv_ps(vf1p0, vx, (__m128)vm);
        vr = _mm_mul_ps(vr, vt);
        vj = _mm_srli_epi32(vj, 1);
        vm = _mm_cmpgt_epi32(vj, vi0);
        if (0 == _mm_movemask_epi8(vm)) break;
        vx = _mm_blendv_ps(vf1p0, vx, (__m128)vm);
        vx = _mm_mul_ps(vx, vx);
    }

    if (0 != _mm_movemask_ps((__m128)vi)) {
        vx = _mm_div_ps(vf1p0, vr);
        vr = _mm_blendv_ps(vr, vx, (__m128)vi);
    }

    return vr;
}

/**
 *  \brief Compute R4(:)**I4(:) under mask
 *  \param[in] _vx (__m128)  R4(:)
 *  \param[in] _vi (__m128i) I4(:)
 *  \param[in] _vm (__m128i) I4(:)
 *  \return (__m128) WHERE(_vm(:) != 0) R4(:)**I4(:)
 */
vrs4_t
__fs_powi_4m_sse4(vrs4_t _vx, vis4_t _vi, vis4_t _vm)
{
    __m128  vx = (__m128)_vx;
    /*
     * Intentionally use PS for integer values, simplifies numer of arguments
     * that need to be cast when using _mm_blendv_ps() intrinsic.
     */
    __m128  vi = (__m128)_vi;
    __m128  vm = (__m128)_vm;

    vx = _mm_blendv_ps(_mm_set1_ps(0.0), vx, vm);
    vi = _mm_blendv_ps((__m128)_mm_set1_epi32(0), vi, vm);

    return __fs_powi_4_sse4((vrs4_t)vx, (vis4_t)vi);
}

/**
 *  \brief Compute R8(:)**I8(:)
 *  \param[in] _vx (__m128d) R8(:)
 *  \param[in] _vi (__m128i) I8(:)
 *  \return (__m128d) R4(:)**I8(:)
 */
vrd2_t
__fd_powk_2_sse4(vrd2_t _vx, vid2_t _vi)
{
    __m128d vx = (__m128d)_vx;
    __m128i vi = (__m128i)_vi;
    __m128d vt;
    __m128d vr;
    __m128i vm;
    __m128i vj;
    __m128d vf1p0 = _mm_set1_pd(1.0);
    __m128i vi0   = _mm_setzero_si128();
    __m128i vlsb  = _mm_set1_epi64x(1);

    vj = _mm_sub_epi64(vi0, vi);
    vm = _mm_cmpgt_epi64(vi0, vi);
    vj = (__m128i)_mm_blendv_pd((__m128d)vi, (__m128d)vj, (__m128d)vm);
    vr = vf1p0;

    vm = _mm_cmpgt_epi64(vj, vi0);
    if (0 == _mm_movemask_epi8(vm)) return vr;

    for (;;) {
        vm = _mm_and_si128(vj, vlsb);
        vm = _mm_sub_epi64(vi0, vm);
        // Where vm == -1, vt = vx, else 1.0
        vt = _mm_blendv_pd(vf1p0, vx, (__m128d)vm);
        vr = _mm_mul_pd(vr, vt);
        vj = _mm_srli_epi64(vj, 1);
        vm = _mm_cmpgt_epi64(vj, vi0);
        if (0 == _mm_movemask_epi8(vm)) break;
        vx = _mm_blendv_pd(vf1p0, vx, (__m128d)vm);
        vx = _mm_mul_pd(vx, vx);
    }

    if (0 != _mm_movemask_pd((__m128d)vi)) {
        vx = _mm_div_pd(vf1p0, vr);
        vr = _mm_blendv_pd(vr, vx, (__m128d)vi);
    }

    return vr;
}

/**
 *  \brief Compute R8(:)**I8(:) under mask
 *  \param[in] _vx (__m128d) R8(:)
 *  \param[in] _vi (__m128i) I8(:)
 *  \param[in] _vm (__m128i) I8(:)
 *  \return (__m128d) WHERE(_vm(:) != 0) R8(:)**I8(:)
 */
vrd2_t
__fd_powk_2m_sse4(vrd2_t _vx, vid2_t _vi, vid2_t _vm)
{
    __m128d vx = (__m128d)_vx;
    /*
     * Intentionally use PD for integer values, simplifies numer of arguments
     * that need to be cast when using _mm_blendv_pd() intrinsic.
     */
    __m128d vi = (__m128d)_vi;
    __m128d vm = (__m128d)_vm;

    vx = _mm_blendv_pd(_mm_set1_pd(0.0), vx, vm);
    vi = _mm_blendv_pd((__m128d)_mm_set1_epi64x(0), vi, vm);

    return __fd_powk_2_sse4((vrd2_t)vx, (vid2_t)vi);
}

/**
 *  \brief Compute R8(:)**I4(:)
 *  \param[in] _vx (__m128d) R8(:)
 *  \param[in] _vi (__m128i) I4(:)
 *  \return (__m128d) R8(:)**I4(:)
 */
vrd2_t
__fd_powi_2_sse4(vrd2_t _vx, vis4_t _vi)
{
    return __fd_powk_2_sse4(_vx, (vid2_t)_mm_cvtepi32_epi64((__m128i)_vi));
}

/**
 *  \brief Compute R8(:)**I8(:) under mask
 *  \param[in] _vx (__m128d) R8(:)
 *  \param[in] _vi (__m128i) I4(:)
 *  \param[in] _vm (__m128i) I8(:)
 *  \return (__m128d) WHERE(_vm(:) != 0) R8(:)**I8(:)
 */
vrd2_t
__fd_powi_2m_sse4(vrd2_t _vx, vis4_t _vi, vid2_t _vm)
{
    return __fd_powk_2m_sse4(_vx, (vid2_t)_mm_cvtepi32_epi64((__m128i)_vi), _vm);
}

/**
 *  \brief (internal) Kernel to compute R4(0:3)**I8_lower(:), Compute R4(4:7)**I8_upper(:)
 *  \param[in] _vx (__m128d) R4(:)
 *  \param[in] _vl (__m128i) I8(:)
 *  \param[in] _vu (__m128i) I8(:)
 *  \return (__m128d) (R4(4:7)**I8_upper(:))<<128 | R4(0:3)**I8_lower(:)
 */
vrs4_t
__fs_powk_2x2_sse4 (vrs4_t _vx, vid2_t _vl, vid2_t _vu)
{
    __m128  vx;
    __m128  vt;
    __m128  vr;
    __m128  vrl;
    __m128  vf1p0 = _mm_set1_ps(1.0);
    __m128i vmi;    // Inner loop mask
    __m128i vmo;    // Outer loop mask
    __m128i vi;
    __m128i vj;
    __m128i vi0 = _mm_setzero_si128();
    __m128i vlsb  = _mm_set1_epi64x(1);

    int     i;

    vx = (__m128)_mm_shuffle_epi32((__m128i)_vx, 0x50);
    vi = (__m128i)_vl;

    for(i = 0 ; i < 2 ; i++) {
        vr  = vf1p0;
        vj  = _mm_sub_epi64(vi0, vi);
        vmo = _mm_cmpgt_epi64(vi0, vi);
        vj  = (__m128i)_mm_blendv_pd((__m128d)vi, (__m128d)vj, (__m128d)vmo);
        for (;;) {
            vmi = _mm_and_si128(vj, vlsb);
            vmi = _mm_sub_epi64(vi0, vmi);
            vt  = (__m128) _mm_blendv_pd((__m128d)vf1p0, (__m128d)vx, (__m128d)vmi);
            vr  = _mm_mul_ps(vr, vt);
            vj  = _mm_srli_epi64(vj, 1);
            vmi = _mm_cmpgt_epi64(vj, vi0);
            if (0 == _mm_movemask_epi8(vmi)) break;
            vx  = (__m128) _mm_blendv_pd((__m128d)vf1p0, (__m128d)vx, (__m128d)vmi);
            vx  = _mm_mul_ps(vx, vx);
        }
        if (0 != _mm_movemask_pd((__m128d)vmo)) {
            vx = _mm_div_ps(vf1p0, vr);
            vr = (__m128) _mm_blendv_pd((__m128d)vr, (__m128d)vx, (__m128d)vmo);
        }
        if (i == 1) break;
        vrl = vr;
        vx = (__m128)_mm_shuffle_epi32((__m128i)_vx, 0xfa);
        vi = (__m128i)_vu;
    }

    return (vrs4_t) _mm_shuffle_ps(vrl, vr, 0x88);
}

/**
 *  \brief (external) Compute R4(0:3)**I8_lower(:), Compute R4(4:7)**I8_upper(:)
 *  \param[in] _vx (__m128d) R4(:)
 *  \param[in] _vl (__m128i) I8(:)
 *  \param[in] _vu (__m128i) I8(:)
 *  \return (__m128d) (R4(4:7)**I8_upper(:))<<128 | R4(0:3)**I8_lower(:)
 */
vrs4_t
__fs_powk_4_sse4(vrs4_t _vx, vid2_t _vl, vid2_t _vu)
{
    __m128i vu = (__m128i) _vu;
    __m128i vl = (__m128i) _vl;
    __m128i vabsu;
    __m128i vabsl;
    __m128i vi0 = _mm_setzero_si128();
    __m128i vi2to31m1 = _mm_set1_epi64x((1ll<<31)-1);
    __m128i vmu;
    __m128i vml;
    __m128i vi;

    vabsl = _mm_sub_epi64(vi0, vl);
    vabsu = _mm_sub_epi64(vi0, vu);
    vml   = _mm_cmpgt_epi64(vi0, vl);
    vmu   = _mm_cmpgt_epi64(vi0, vu);
    vabsl = (__m128i)_mm_blendv_pd((__m128d)vl, (__m128d)vabsl, (__m128d)vml);
    vabsu = (__m128i)_mm_blendv_pd((__m128d)vu, (__m128d)vabsu, (__m128d)vmu);

    vml = _mm_cmpgt_epi64(vabsl, vi2to31m1);
    vmu = _mm_cmpgt_epi64(vabsu, vi2to31m1);

    if (0 != _mm_movemask_epi8(vmu|vml)) return __fs_powk_2x2_sse4(_vx, _vl, _vu);

    vabsl = _mm_shuffle_epi32(vl, 0x08);
    vabsu = _mm_shuffle_epi32(vu, 0x80);

    vi = (__m128i)_mm_blend_ps((__m128)vabsu, (__m128)vabsl, 0x3);
    return __fs_powi_4_sse4(_vx, (vis4_t) vi);

}

/**
 *  \brief (external) Compute R4(0:3)**I8_lower(:), Compute R4(4:7)**I8_upper(:)
 *  \param[in] _vx (__m128d) R4(:)
 *  \param[in] _vl (__m128i) I8(:)
 *  \param[in] _vu (__m128i) I8(:)
 *  \param[in] _vm (__m128i) I8(:)
 *  \return (__m128d) (R4(4:7)**I8_upper(:))<<128 | R4(0:3)**I8_lower(:)
 */
vrs4_t
__fs_powk_4m_sse4(vrs4_t _vx, vid2_t _vl, vid2_t _vu, vis4_t _vm)
{
    __m128  vx = (__m128) _vx;
    /*
     * Intentionally use PD for integer values, simplifies numer of arguments
     * that need to be cast when using _mm_blendv_pd() intrinsic.
     */
    __m128d vu = (__m128d) _vu;
    __m128d vl = (__m128d) _vl;
    __m128d vt;

    vx = _mm_blendv_ps(_mm_set1_ps(0.0), vx, (__m128)_vm);
    vt = (__m128d)_mm_cvtepi32_epi64((__m128i)_vm);
    vl = _mm_blendv_pd(_mm_set1_pd(0), vl, vt);
    vt = (__m128d)_mm_cvtepi32_epi64((__m128i)_mm_srli_si128((__m128i)_vm, 8));
    vu = _mm_blendv_pd(_mm_set1_pd(0), vu, vt);
    return __fs_powk_4_sse4((vrs4_t)vx, (vid2_t)vl, (vid2_t)vu);
}
