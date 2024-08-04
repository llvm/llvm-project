/*===---- avx10_2niintrin.h - AVX10.2 new instruction intrinsics -----------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error "Never use <avx10_2niintrin.h> directly; include <immintrin.h> instead."
#endif

#ifdef __SSE2__

#ifndef __AVX10_2NIINTRIN_H
#define __AVX10_2NIINTRIN_H

/* VMPSADBW */
#define _mm_mask_mpsadbw_epu8(W, U, A, B, imm)                                 \
  ((__m128i)__builtin_ia32_selectw_128(                                        \
      (__mmask8)(U), (__v8hi)_mm_mpsadbw_epu8((A), (B), (imm)),                \
      (__v8hi)(__m128i)(W)))

#define _mm_maskz_mpsadbw_epu8(U, A, B, imm)                                   \
  ((__m128i)__builtin_ia32_selectw_128(                                        \
      (__mmask8)(U), (__v8hi)_mm_mpsadbw_epu8((A), (B), (imm)),                \
      (__v8hi)_mm_setzero_si128()))

#define _mm256_mask_mpsadbw_epu8(W, U, A, B, imm)                              \
  ((__m256i)__builtin_ia32_selectw_256(                                        \
      (__mmask16)(U), (__v16hi)_mm256_mpsadbw_epu8((A), (B), (imm)),           \
      (__v16hi)(__m256i)(W)))

#define _mm256_maskz_mpsadbw_epu8(U, A, B, imm)                                \
  ((__m256i)__builtin_ia32_selectw_256(                                        \
      (__mmask16)(U), (__v16hi)_mm256_mpsadbw_epu8((A), (B), (imm)),           \
      (__v16hi)_mm256_setzero_si256()))

/* YMM Rounding */
#define _mm256_add_round_pd(A, B, R)                                           \
  ((__m256d)__builtin_ia32_vaddpd256_round((__v4df)(__m256d)(A),               \
                                           (__v4df)(__m256d)(B), (int)(R)))

#define _mm256_mask_add_round_pd(W, U, A, B, R)                                \
  ((__m256d)__builtin_ia32_selectpd_256(                                       \
      (__mmask8)(U), (__v4df)_mm256_add_round_pd((A), (B), (R)),               \
      (__v4df)(__m256d)(W)))

#define _mm256_maskz_add_round_pd(U, A, B, R)                                  \
  ((__m256d)__builtin_ia32_selectpd_256(                                       \
      (__mmask8)(U), (__v4df)_mm256_add_round_pd((A), (B), (R)),               \
      (__v4df)_mm256_setzero_pd()))

#define _mm256_add_round_ph(A, B, R)                                           \
  ((__m256h)__builtin_ia32_vaddph256_round((__v16hf)(__m256h)(A),              \
                                           (__v16hf)(__m256h)(B), (int)(R)))

#define _mm256_mask_add_round_ph(W, U, A, B, R)                                \
  ((__m256h)__builtin_ia32_selectph_256(                                       \
      (__mmask16)(U), (__v16hf)_mm256_add_round_ph((A), (B), (R)),             \
      (__v16hf)(__m256h)(W)))

#define _mm256_maskz_add_round_ph(U, A, B, R)                                  \
  ((__m256h)__builtin_ia32_selectph_256(                                       \
      (__mmask16)(U), (__v16hf)_mm256_add_round_ph((A), (B), (R)),             \
      (__v16hf)_mm256_setzero_ph()))

#define _mm256_add_round_ps(A, B, R)                                           \
  ((__m256)__builtin_ia32_vaddps256_round((__v8sf)(__m256)(A),                 \
                                          (__v8sf)(__m256)(B), (int)(R)))

#define _mm256_mask_add_round_ps(W, U, A, B, R)                                \
  ((__m256)__builtin_ia32_selectps_256(                                        \
      (__mmask8)(U), (__v8sf)_mm256_add_round_ps((A), (B), (R)),               \
      (__v8sf)(__m256)(W)))

#define _mm256_maskz_add_round_ps(U, A, B, R)                                  \
  ((__m256)__builtin_ia32_selectps_256(                                        \
      (__mmask8)(U), (__v8sf)_mm256_add_round_ps((A), (B), (R)),               \
      (__v8sf)_mm256_setzero_ps()))

#endif /* __AVX10_2NIINTRIN_H */
#endif /* __SSE2__ */
