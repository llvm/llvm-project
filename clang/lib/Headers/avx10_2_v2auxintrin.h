/*===------------ avx10_2_v2auxintrin.h - AVX10_2_V2AUX -------------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <avx10_2_v2auxintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifdef __SSE2__

#ifndef __AVX10_2_V2AUXINTRIN_H
#define __AVX10_2_V2AUXINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10-v2-aux"),   \
                 __min_vector_width__(128)))
#define __DEFAULT_FN_ATTRS256                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10-v2-aux"),   \
                 __min_vector_width__(256)))
#define __DEFAULT_FN_ATTRS512                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10-v2-aux"),   \
                 __min_vector_width__(512)))

// clang-format off

//===----------------------------------------------------------------------===//
// Group A: VCVTPS2BF8 / VCVTPS2BF8S / VCVTPS2HF8 / VCVTPS2HF8S /
//          VCVTROPS2HF8 / VCVTROPS2HF8S
// Convert packed single-precision to FP8. Output is always __m128i.
//===----------------------------------------------------------------------===//

// VCVTPS2BF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtps_bf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_128_mask(
      (__v4sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtps_bf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtps_bf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2BF8 - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtps_bf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_256_mask(
      (__v8sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtps_bf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2BF8 - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtps_bf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_512_mask(
      (__v16sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtps_bf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask16)__U);
}

// VCVTPS2BF8S - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvts_ps_bf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_128_mask(
      (__v4sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_ps_bf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2BF8S - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_ps_bf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_256_mask(
      (__v8sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvts_ps_bf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_ps_bf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2BF8S - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_ps_bf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_512_mask(
      (__v16sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_ps_bf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_ps_bf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2bf8s_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask16)__U);
}

// VCVTPS2HF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtps_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_128_mask(
      (__v4sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtps_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtps_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2HF8 - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtps_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_256_mask(
      (__v8sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtps_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2HF8 - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtps_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_512_mask(
      (__v16sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtps_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask16)__U);
}

// VCVTPS2HF8S - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvts_ps_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_128_mask(
      (__v4sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_ps_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2HF8S - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_ps_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_256_mask(
      (__v8sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvts_ps_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_ps_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTPS2HF8S - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_ps_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_512_mask(
      (__v16sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_ps_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_ps_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtps2hf8s_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask16)__U);
}

// VCVTROPS2HF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtrops_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_128_mask(
      (__v4sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtrops_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTROPS2HF8 - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtrops_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_256_mask(
      (__v8sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtrops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtrops_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTROPS2HF8 - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtrops_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_512_mask(
      (__v16sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtrops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtrops_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask16)__U);
}

// VCVTROPS2HF8S - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvts_rops_hf8(__m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_128_mask(
      (__v4sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_rops_hf8(__mmask8 __U, __m128 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_128_mask(
      (__v4sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTROPS2HF8S - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_rops_hf8(__m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_256_mask(
      (__v8sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvts_rops_hf8(__m128i __W, __mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_rops_hf8(__mmask8 __U, __m256 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_256_mask(
      (__v8sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask8)__U);
}

// VCVTROPS2HF8S - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_rops_hf8(__m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_512_mask(
      (__v16sf)__A, (__v16qi)_mm_undefined_si128(), (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_rops_hf8(__m128i __W, __mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_rops_hf8(__mmask16 __U, __m512 __A) {
  return (__m128i)__builtin_ia32_vcvtrops2hf8s_512_mask(
      (__v16sf)__A, (__v16qi)(__m128i)_mm_setzero_si128(), (__mmask16)__U);
}

//===----------------------------------------------------------------------===//
// Group B: VCVTBIASPS2BF8 / VCVTBIASPS2BF8S / VCVTBIASPS2HF8 /
//          VCVTBIASPS2HF8S
// Convert packed single-precision with bias to FP8. Output is always __m128i.
//===----------------------------------------------------------------------===//

// VCVTBIASPS2BF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtbiasps_bf8(__m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbiasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2BF8 - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtbiasps_bf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbiasps_bf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbiasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2BF8 - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtbiasps_bf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)_mm_undefined_si128(),
      (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbiasps_bf8(__m128i __W, __mmask16 __U, __m512i __A,
                          __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbiasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask16)__U);
}

// VCVTBIASPS2BF8S - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvts_biasps_bf8(__m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_biasps_bf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_biasps_bf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2BF8S - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_biasps_bf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_mask_cvts_biasps_bf8(
    __m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_biasps_bf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2BF8S - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_biasps_bf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)_mm_undefined_si128(),
      (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_biasps_bf8(__m128i __W, __mmask16 __U, __m512i __A,
                           __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_biasps_bf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2bf8s_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask16)__U);
}

// VCVTBIASPS2HF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtbiasps_hf8(__m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbiasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2HF8 - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvtbiasps_hf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbiasps_hf8(__m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbiasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2HF8 - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvtbiasps_hf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)_mm_undefined_si128(),
      (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbiasps_hf8(__m128i __W, __mmask16 __U, __m512i __A,
                          __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbiasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask16)__U);
}

// VCVTBIASPS2HF8S - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvts_biasps_hf8(__m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvts_biasps_hf8(__m128i __W, __mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvts_biasps_hf8(__mmask8 __U, __m128i __A, __m128 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_128_mask(
      (__v16qi)__A, (__v4sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2HF8S - 256-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_cvts_biasps_hf8(__m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)_mm_undefined_si128(), (__mmask8)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256 _mm256_mask_cvts_biasps_hf8(
    __m128i __W, __mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)__W, (__mmask8)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvts_biasps_hf8(__mmask8 __U, __m256i __A, __m256 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_256_mask(
      (__v32qi)__A, (__v8sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask8)__U);
}

// VCVTBIASPS2HF8S - 512-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_cvts_biasps_hf8(__m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)_mm_undefined_si128(),
      (__mmask16)-1);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_mask_cvts_biasps_hf8(__m128i __W, __mmask16 __U, __m512i __A,
                           __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)__W, (__mmask16)__U);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvts_biasps_hf8(__mmask16 __U, __m512i __A, __m512 __B) {
  return (__m128i)__builtin_ia32_vcvtbiasps2hf8s_512_mask(
      (__v64qi)__A, (__v16sf)__B, (__v16qi)(__m128i)_mm_setzero_si128(),
      (__mmask16)__U);
}

//===----------------------------------------------------------------------===//
// Group C: VCVTBF82PS / VCVTHF82PS
// Convert packed FP8 to single-precision. Input is __m128i, output varies.
//===----------------------------------------------------------------------===//

// VCVTBF82PS - 128-bit

static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_cvtbf8_ps(__m128i __A) {
  return (__m128)__builtin_ia32_vcvtbf8_2ps128_mask(
      (__v16qi)__A, (__v4sf)_mm_undefined_ps(), (__mmask8)-1);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_mask_cvtbf8_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  return (__m128)__builtin_ia32_vcvtbf8_2ps128_mask(
      (__v16qi)__A, (__v4sf)__W, (__mmask8)__U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  return (__m128)__builtin_ia32_vcvtbf8_2ps128_mask(
      (__v16qi)__A, (__v4sf)_mm_setzero_ps(), (__mmask8)__U);
}

// VCVTBF82PS - 256-bit

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_cvtbf8_ps(__m128i __A) {
  return (__m256)__builtin_ia32_vcvtbf8_2ps256_mask(
      (__v16qi)__A, (__v8sf)_mm256_undefined_ps(), (__mmask8)-1);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_vcvtbf8_2ps256_mask(
      (__v16qi)__A, (__v8sf)__W, (__mmask8)__U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbf8_ps(__mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_vcvtbf8_2ps256_mask(
      (__v16qi)__A, (__v8sf)_mm256_setzero_ps(), (__mmask8)__U);
}

// VCVTBF82PS - 512-bit

static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_cvtbf8_ps(__m128i __A) {
  return (__m512)__builtin_ia32_vcvtbf8_2ps512_mask(
      (__v16qi)__A, (__v16sf)_mm512_undefined_ps(), (__mmask16)-1);
}

static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_vcvtbf8_2ps512_mask(
      (__v16qi)__A, (__v16sf)__W, (__mmask16)__U);
}

static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbf8_ps(__mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_vcvtbf8_2ps512_mask(
      (__v16qi)__A, (__v16sf)_mm512_setzero_ps(), (__mmask16)__U);
}

// VCVTHF82PS - 128-bit

static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_cvthf8_ps(__m128i __A) {
  return (__m128)__builtin_ia32_vcvthf8_2ps128_mask(
      (__v16qi)__A, (__v4sf)_mm_undefined_ps(), (__mmask8)-1);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_mask_cvthf8_ps(__m128 __W, __mmask8 __U, __m128i __A) {
  return (__m128)__builtin_ia32_vcvthf8_2ps128_mask(
      (__v16qi)__A, (__v4sf)__W, (__mmask8)__U);
}

static __inline__ __m128 __DEFAULT_FN_ATTRS128
_mm_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  return (__m128)__builtin_ia32_vcvthf8_2ps128_mask(
      (__v16qi)__A, (__v4sf)_mm_setzero_ps(), (__mmask8)__U);
}

// VCVTHF82PS - 256-bit

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_cvthf8_ps(__m128i __A) {
  return (__m256)__builtin_ia32_vcvthf8_2ps256_mask(
      (__v16qi)__A, (__v8sf)_mm256_undefined_ps(), (__mmask8)-1);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_mask_cvthf8_ps(__m256 __W, __mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_vcvthf8_2ps256_mask(
      (__v16qi)__A, (__v8sf)__W, (__mmask8)__U);
}

static __inline__ __m256 __DEFAULT_FN_ATTRS256
_mm256_maskz_cvthf8_ps(__mmask8 __U, __m128i __A) {
  return (__m256)__builtin_ia32_vcvthf8_2ps256_mask(
      (__v16qi)__A, (__v8sf)_mm256_setzero_ps(), (__mmask8)__U);
}

// VCVTHF82PS - 512-bit

static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_cvthf8_ps(__m128i __A) {
  return (__m512)__builtin_ia32_vcvthf8_2ps512_mask(
      (__v16qi)__A, (__v16sf)_mm512_undefined_ps(), (__mmask16)-1);
}

static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_mask_cvthf8_ps(__m512 __W, __mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_vcvthf8_2ps512_mask(
      (__v16qi)__A, (__v16sf)__W, (__mmask16)__U);
}

static __inline__ __m512 __DEFAULT_FN_ATTRS512
_mm512_maskz_cvthf8_ps(__mmask16 __U, __m128i __A) {
  return (__m512)__builtin_ia32_vcvthf8_2ps512_mask(
      (__v16qi)__A, (__v16sf)_mm512_setzero_ps(), (__mmask16)__U);
}

//===----------------------------------------------------------------------===//
// Group E: VCVTBF82BF6S / VCVTHF82HF6S
// Same-size reg-only conversions (no masking support)
//===----------------------------------------------------------------------===//

// VCVTBF82BF6S

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtbf8_bf6s(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvtbf82bf6s128((__v16qi)__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtbf8_bf6s(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvtbf82bf6s256((__v32qi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_cvtbf8_bf6s(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvtbf82bf6s512((__v64qi)__A);
}

// VCVTHF82HF6S

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvthf8_hf6s(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvthf82hf6s128((__v16qi)__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvthf8_hf6s(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvthf82hf6s256((__v32qi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_cvthf8_hf6s(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvthf82hf6s512((__v64qi)__A);
}

//===----------------------------------------------------------------------===//
// Group F: VCVTBF42HF8 / VCVTBF62HF8 / VCVTHF62HF8
// Expanding/same-size conversions with masking support
//===----------------------------------------------------------------------===//

// VCVTBF42HF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtbf4_hf8(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvtbf42hf8128((__v16qi)__A);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbf4_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      __U, (__v16qi)_mm_cvtbf4_hf8(__A), (__v16qi)__W);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbf4_hf8(__mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      __U, (__v16qi)_mm_cvtbf4_hf8(__A), (__v16qi)_mm_setzero_si128());
}

// VCVTBF42HF8 - 256-bit

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtbf4_hf8(__m128i __A) {
  return (__m256i)__builtin_ia32_vcvtbf42hf8256((__v16qi)__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbf4_hf8(__m256i __W, __mmask32 __U, __m128i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf4_hf8(__A), (__v32qi)__W);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbf4_hf8(__mmask32 __U, __m128i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf4_hf8(__A), (__v32qi)_mm256_setzero_si256());
}

// VCVTBF42HF8 - 512-bit

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_cvtbf4_hf8(__m256i __A) {
  return (__m512i)__builtin_ia32_vcvtbf42hf8512((__v32qi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbf4_hf8(__m512i __W, __mmask64 __U, __m256i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf4_hf8(__A), (__v64qi)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbf4_hf8(__mmask64 __U, __m256i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf4_hf8(__A), (__v64qi)_mm512_setzero_si512());
}

// VCVTBF62HF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvtbf6_hf8(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvtbf62hf8128((__v16qi)__A);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvtbf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      __U, (__v16qi)_mm_cvtbf6_hf8(__A), (__v16qi)__W);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvtbf6_hf8(__mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      __U, (__v16qi)_mm_cvtbf6_hf8(__A), (__v16qi)_mm_setzero_si128());
}

// VCVTBF62HF8 - 256-bit

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvtbf6_hf8(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvtbf62hf8256((__v32qi)__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_cvtbf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf6_hf8(__A), (__v32qi)__W);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvtbf6_hf8(__mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvtbf6_hf8(__A), (__v32qi)_mm256_setzero_si256());
}

// VCVTBF62HF8 - 512-bit

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_cvtbf6_hf8(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvtbf62hf8512((__v64qi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_cvtbf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf6_hf8(__A), (__v64qi)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvtbf6_hf8(__mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvtbf6_hf8(__A), (__v64qi)_mm512_setzero_si512());
}

// VCVTHF62HF8 - 128-bit

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_cvthf6_hf8(__m128i __A) {
  return (__m128i)__builtin_ia32_vcvthf62hf8128((__v16qi)__A);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_mask_cvthf6_hf8(__m128i __W, __mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      __U, (__v16qi)_mm_cvthf6_hf8(__A), (__v16qi)__W);
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128
_mm_maskz_cvthf6_hf8(__mmask16 __U, __m128i __A) {
  return (__m128i)__builtin_ia32_selectb_128(
      __U, (__v16qi)_mm_cvthf6_hf8(__A), (__v16qi)_mm_setzero_si128());
}

// VCVTHF62HF8 - 256-bit

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_cvthf6_hf8(__m256i __A) {
  return (__m256i)__builtin_ia32_vcvthf62hf8256((__v32qi)__A);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_mask_cvthf6_hf8(__m256i __W, __mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvthf6_hf8(__A), (__v32qi)__W);
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_maskz_cvthf6_hf8(__mmask32 __U, __m256i __A) {
  return (__m256i)__builtin_ia32_selectb_256(
      __U, (__v32qi)_mm256_cvthf6_hf8(__A), (__v32qi)_mm256_setzero_si256());
}

// VCVTHF62HF8 - 512-bit

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_cvthf6_hf8(__m512i __A) {
  return (__m512i)__builtin_ia32_vcvthf62hf8512((__v64qi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_mask_cvthf6_hf8(__m512i __W, __mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvthf6_hf8(__A), (__v64qi)__W);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_maskz_cvthf6_hf8(__mmask64 __U, __m512i __A) {
  return (__m512i)__builtin_ia32_selectb_512(
      __U, (__v64qi)_mm512_cvthf6_hf8(__A), (__v64qi)_mm512_setzero_si512());
}

//===----------------------------------------------------------------------===//
// Group H: VUNPACKB
// Byte unpack with immediate
//===----------------------------------------------------------------------===//

// VUNPACKB - 128-bit

#define _mm_unpackb_epi8(A, imm)                                               \
  ((__m128i)__builtin_ia32_vunpackb128((__v16qi)(__m128i)(A), (int)(imm)))

#define _mm_mask_unpackb_epi8(W, U, A, imm)                                    \
  ((__m128i)__builtin_ia32_selectb_128(                                         \
      (__mmask16)(U),                                                           \
      (__v16qi)_mm_unpackb_epi8((A), (imm)),                                    \
      (__v16qi)(__m128i)(W)))

#define _mm_maskz_unpackb_epi8(U, A, imm)                                      \
  ((__m128i)__builtin_ia32_selectb_128(                                         \
      (__mmask16)(U),                                                           \
      (__v16qi)_mm_unpackb_epi8((A), (imm)),                                    \
      (__v16qi)_mm_setzero_si128()))

// VUNPACKB - 256-bit

#define _mm256_unpackb_epi8(A, imm)                                            \
  ((__m256i)__builtin_ia32_vunpackb256((__v32qi)(__m256i)(A), (int)(imm)))

#define _mm256_mask_unpackb_epi8(W, U, A, imm)                                 \
  ((__m256i)__builtin_ia32_selectb_256(                                         \
      (__mmask32)(U),                                                           \
      (__v32qi)_mm256_unpackb_epi8((A), (imm)),                                 \
      (__v32qi)(__m256i)(W)))

#define _mm256_maskz_unpackb_epi8(U, A, imm)                                   \
  ((__m256i)__builtin_ia32_selectb_256(                                         \
      (__mmask32)(U),                                                           \
      (__v32qi)_mm256_unpackb_epi8((A), (imm)),                                 \
      (__v32qi)_mm256_setzero_si256()))

// VUNPACKB - 512-bit

#define _mm512_unpackb_epi8(A, imm)                                            \
  ((__m512i)__builtin_ia32_vunpackb512((__v64qi)(__m512i)(A), (int)(imm)))

#define _mm512_mask_unpackb_epi8(W, U, A, imm)                                 \
  ((__m512i)__builtin_ia32_selectb_512(                                         \
      (__mmask64)(U),                                                           \
      (__v64qi)_mm512_unpackb_epi8((A), (imm)),                                 \
      (__v64qi)(__m512i)(W)))

#define _mm512_maskz_unpackb_epi8(U, A, imm)                                   \
  ((__m512i)__builtin_ia32_selectb_512(                                         \
      (__mmask64)(U),                                                           \
      (__v64qi)_mm512_unpackb_epi8((A), (imm)),                                 \
      (__v64qi)_mm512_setzero_si512()))

// clang-format on

#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256
#undef __DEFAULT_FN_ATTRS512

#endif // __AVX10_2_V2AUXINTRIN_H
#endif // __SSE2__
