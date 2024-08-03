/*===---- avx10_2_512niintrin.h - AVX10.2-512 new instruction intrinsics ---===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <avx10_2_512niintrin.h> directly; include <immintrin.h> instead."
#endif

#ifdef __SSE2__

#ifndef __AVX10_2_512NIINTRIN_H
#define __AVX10_2_512NIINTRIN_H

/* VMPSADBW */
#define _mm512_mpsadbw_epu8(A, B, imm)                                         \
  ((__m512i)__builtin_ia32_mpsadbw512((__v64qi)(__m512i)(A),                   \
                                      (__v64qi)(__m512i)(B), (int)(imm)))

#define _mm512_mask_mpsadbw_epu8(W, U, A, B, imm)                              \
  ((__m512i)__builtin_ia32_selectw_512(                                        \
      (__mmask32)(U), (__v32hi)_mm512_mpsadbw_epu8((A), (B), (imm)),           \
      (__v32hi)(__m512i)(W)))

#define _mm512_maskz_mpsadbw_epu8(U, A, B, imm)                                \
  ((__m512i)__builtin_ia32_selectw_512(                                        \
      (__mmask32)(U), (__v32hi)_mm512_mpsadbw_epu8((A), (B), (imm)),           \
      (__v32hi)_mm512_setzero_si512()))

#endif /* __SSE2__ */
#endif /* __AVX10_2_512NIINTRIN_H */
