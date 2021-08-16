/*===----------- avx512fp16intrin.h - AVX512-FP16 intrinsics ---------------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error "Never use <avx512fp16intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512FP16INTRIN_H
#define __AVX512FP16INTRIN_H

/* Define the default attributes for the functions in this file. */
typedef _Float16 __v32hf __attribute__((__vector_size__(64), __aligned__(64)));
typedef _Float16 __m512h __attribute__((__vector_size__(64), __aligned__(64)));
typedef _Float16 __m512h_u __attribute__((__vector_size__(64), __aligned__(1)));
typedef _Float16 __v8hf __attribute__((__vector_size__(16), __aligned__(16)));
typedef _Float16 __m128h __attribute__((__vector_size__(16), __aligned__(16)));
typedef _Float16 __m128h_u __attribute__((__vector_size__(16), __aligned__(1)));
typedef _Float16 __v16hf __attribute__((__vector_size__(32), __aligned__(32)));
typedef _Float16 __m256h __attribute__((__vector_size__(32), __aligned__(32)));
typedef _Float16 __m256h_u __attribute__((__vector_size__(32), __aligned__(1)));

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS512                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx512fp16"),     \
                 __min_vector_width__(512)))
#define __DEFAULT_FN_ATTRS256                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx512fp16"),     \
                 __min_vector_width__(256)))
#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx512fp16"),     \
                 __min_vector_width__(128)))

static __inline__ _Float16 __DEFAULT_FN_ATTRS512 _mm512_cvtsh_h(__m512h __a) {
  return __a[0];
}

static __inline __m128h __DEFAULT_FN_ATTRS128 _mm_setzero_ph(void) {
  return (__m128h){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

static __inline __m256h __DEFAULT_FN_ATTRS256 _mm256_setzero_ph(void) {
  return (__m256h){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256 _mm256_undefined_ph(void) {
  return (__m256h)__builtin_ia32_undef256();
}

static __inline __m512h __DEFAULT_FN_ATTRS512 _mm512_setzero_ph(void) {
  return (__m512h){0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_undefined_ph(void) {
  return (__m128h)__builtin_ia32_undef128();
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_undefined_ph(void) {
  return (__m512h)__builtin_ia32_undef512();
}

static __inline __m512h __DEFAULT_FN_ATTRS512 _mm512_set1_ph(_Float16 __h) {
  return (__m512h)(__v32hf){__h, __h, __h, __h, __h, __h, __h, __h,
                            __h, __h, __h, __h, __h, __h, __h, __h,
                            __h, __h, __h, __h, __h, __h, __h, __h,
                            __h, __h, __h, __h, __h, __h, __h, __h};
}

static __inline __m512h __DEFAULT_FN_ATTRS512
_mm512_set_ph(_Float16 __h1, _Float16 __h2, _Float16 __h3, _Float16 __h4,
              _Float16 __h5, _Float16 __h6, _Float16 __h7, _Float16 __h8,
              _Float16 __h9, _Float16 __h10, _Float16 __h11, _Float16 __h12,
              _Float16 __h13, _Float16 __h14, _Float16 __h15, _Float16 __h16,
              _Float16 __h17, _Float16 __h18, _Float16 __h19, _Float16 __h20,
              _Float16 __h21, _Float16 __h22, _Float16 __h23, _Float16 __h24,
              _Float16 __h25, _Float16 __h26, _Float16 __h27, _Float16 __h28,
              _Float16 __h29, _Float16 __h30, _Float16 __h31, _Float16 __h32) {
  return (__m512h)(__v32hf){__h32, __h31, __h30, __h29, __h28, __h27, __h26,
                            __h25, __h24, __h23, __h22, __h21, __h20, __h19,
                            __h18, __h17, __h16, __h15, __h14, __h13, __h12,
                            __h11, __h10, __h9,  __h8,  __h7,  __h6,  __h5,
                            __h4,  __h3,  __h2,  __h1};
}

#define _mm512_setr_ph(h1, h2, h3, h4, h5, h6, h7, h8, h9, h10, h11, h12, h13, \
                       h14, h15, h16, h17, h18, h19, h20, h21, h22, h23, h24,  \
                       h25, h26, h27, h28, h29, h30, h31, h32)                 \
  _mm512_set_ph((h32), (h31), (h30), (h29), (h28), (h27), (h26), (h25), (h24), \
                (h23), (h22), (h21), (h20), (h19), (h18), (h17), (h16), (h15), \
                (h14), (h13), (h12), (h11), (h10), (h9), (h8), (h7), (h6),     \
                (h5), (h4), (h3), (h2), (h1))

static __inline__ __m128 __DEFAULT_FN_ATTRS128 _mm_castph_ps(__m128h __a) {
  return (__m128)__a;
}

static __inline__ __m256 __DEFAULT_FN_ATTRS256 _mm256_castph_ps(__m256h __a) {
  return (__m256)__a;
}

static __inline__ __m512 __DEFAULT_FN_ATTRS512 _mm512_castph_ps(__m512h __a) {
  return (__m512)__a;
}

static __inline__ __m128d __DEFAULT_FN_ATTRS128 _mm_castph_pd(__m128h __a) {
  return (__m128d)__a;
}

static __inline__ __m256d __DEFAULT_FN_ATTRS256 _mm256_castph_pd(__m256h __a) {
  return (__m256d)__a;
}

static __inline__ __m512d __DEFAULT_FN_ATTRS512 _mm512_castph_pd(__m512h __a) {
  return (__m512d)__a;
}

static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_castph_si128(__m128h __a) {
  return (__m128i)__a;
}

static __inline__ __m256i __DEFAULT_FN_ATTRS256
_mm256_castph_si256(__m256h __a) {
  return (__m256i)__a;
}

static __inline__ __m512i __DEFAULT_FN_ATTRS512
_mm512_castph_si512(__m512h __a) {
  return (__m512i)__a;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_castps_ph(__m128 __a) {
  return (__m128h)__a;
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256 _mm256_castps_ph(__m256 __a) {
  return (__m256h)__a;
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_castps_ph(__m512 __a) {
  return (__m512h)__a;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_castpd_ph(__m128d __a) {
  return (__m128h)__a;
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256 _mm256_castpd_ph(__m256d __a) {
  return (__m256h)__a;
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_castpd_ph(__m512d __a) {
  return (__m512h)__a;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_castsi128_ph(__m128i __a) {
  return (__m128h)__a;
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_castsi256_ph(__m256i __a) {
  return (__m256h)__a;
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_castsi512_ph(__m512i __a) {
  return (__m512h)__a;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS256
_mm256_castph256_ph128(__m256h __a) {
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, 4, 5, 6, 7);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS512
_mm512_castph512_ph128(__m512h __a) {
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, 4, 5, 6, 7);
}

static __inline__ __m256h __DEFAULT_FN_ATTRS512
_mm512_castph512_ph256(__m512h __a) {
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                 12, 13, 14, 15);
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_castph128_ph256(__m128h __a) {
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1,
                                 -1, -1, -1, -1, -1);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_castph128_ph512(__m128h __a) {
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1, -1);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_castph256_ph512(__m256h __a) {
  return __builtin_shufflevector(__a, __a, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                 12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1, -1,
                                 -1, -1, -1, -1, -1, -1, -1, -1);
}

/// Constructs a 256-bit floating-point vector of [16 x half] from a
///    128-bit floating-point vector of [8 x half]. The lower 128 bits
///    contain the value of the source vector. The upper 384 bits are set
///    to zero.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic has no corresponding instruction.
///
/// \param __a
///    A 128-bit vector of [8 x half].
/// \returns A 512-bit floating-point vector of [16 x half]. The lower 128 bits
///    contain the value of the parameter. The upper 384 bits are set to zero.
static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_zextph128_ph256(__m128h __a) {
  return __builtin_shufflevector(__a, (__v8hf)_mm_setzero_ph(), 0, 1, 2, 3, 4,
                                 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}

/// Constructs a 512-bit floating-point vector of [32 x half] from a
///    128-bit floating-point vector of [8 x half]. The lower 128 bits
///    contain the value of the source vector. The upper 384 bits are set
///    to zero.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic has no corresponding instruction.
///
/// \param __a
///    A 128-bit vector of [8 x half].
/// \returns A 512-bit floating-point vector of [32 x half]. The lower 128 bits
///    contain the value of the parameter. The upper 384 bits are set to zero.
static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_zextph128_ph512(__m128h __a) {
  return __builtin_shufflevector(
      __a, (__v8hf)_mm_setzero_ph(), 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
      13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15, 8, 9, 10, 11, 12, 13, 14, 15);
}

/// Constructs a 512-bit floating-point vector of [32 x half] from a
///    256-bit floating-point vector of [16 x half]. The lower 256 bits
///    contain the value of the source vector. The upper 256 bits are set
///    to zero.
///
/// \headerfile <x86intrin.h>
///
/// This intrinsic has no corresponding instruction.
///
/// \param __a
///    A 256-bit vector of [16 x half].
/// \returns A 512-bit floating-point vector of [32 x half]. The lower 256 bits
///    contain the value of the parameter. The upper 256 bits are set to zero.
static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_zextph256_ph512(__m256h __a) {
  return __builtin_shufflevector(__a, (__v16hf)_mm256_setzero_ph(), 0, 1, 2, 3,
                                 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                                 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
                                 29, 30, 31);
}

#define _mm_comi_round_sh(A, B, P, R)                                          \
  __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, (int)(P), (int)(R))

#define _mm_comi_sh(A, B, pred)                                                \
  _mm_comi_round_sh((A), (B), (pred), _MM_FROUND_CUR_DIRECTION)

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_comieq_sh(__m128h A,
                                                          __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_EQ_OS,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_comilt_sh(__m128h A,
                                                          __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_LT_OS,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_comile_sh(__m128h A,
                                                          __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_LE_OS,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_comigt_sh(__m128h A,
                                                          __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_GT_OS,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_comige_sh(__m128h A,
                                                          __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_GE_OS,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_comineq_sh(__m128h A,
                                                           __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_NEQ_US,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_ucomieq_sh(__m128h A,
                                                           __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_EQ_OQ,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_ucomilt_sh(__m128h A,
                                                           __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_LT_OQ,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_ucomile_sh(__m128h A,
                                                           __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_LE_OQ,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_ucomigt_sh(__m128h A,
                                                           __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_GT_OQ,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_ucomige_sh(__m128h A,
                                                           __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_GE_OQ,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ int __DEFAULT_FN_ATTRS128 _mm_ucomineq_sh(__m128h A,
                                                            __m128h B) {
  return __builtin_ia32_vcomish((__v8hf)A, (__v8hf)B, _CMP_NEQ_UQ,
                                _MM_FROUND_CUR_DIRECTION);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_add_ph(__m512h __A,
                                                              __m512h __B) {
  return (__m512h)((__v32hf)__A + (__v32hf)__B);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_mask_add_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512(
      (__mmask32)__U, (__v32hf)_mm512_add_ph(__A, __B), (__v32hf)__W);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_maskz_add_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512((__mmask32)__U,
                                              (__v32hf)_mm512_add_ph(__A, __B),
                                              (__v32hf)_mm512_setzero_ph());
}

#define _mm512_add_round_ph(A, B, R)                                           \
  ((__m512h)__builtin_ia32_addph512((__v32hf)(__m512h)(A),                     \
                                    (__v32hf)(__m512h)(B), (int)(R)))

#define _mm512_mask_add_round_ph(W, U, A, B, R)                                \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_add_round_ph((A), (B), (R)),             \
      (__v32hf)(__m512h)(W)))

#define _mm512_maskz_add_round_ph(U, A, B, R)                                  \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_add_round_ph((A), (B), (R)),             \
      (__v32hf)_mm512_setzero_ph()))

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_sub_ph(__m512h __A,
                                                              __m512h __B) {
  return (__m512h)((__v32hf)__A - (__v32hf)__B);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_mask_sub_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512(
      (__mmask32)__U, (__v32hf)_mm512_sub_ph(__A, __B), (__v32hf)__W);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_maskz_sub_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512((__mmask32)__U,
                                              (__v32hf)_mm512_sub_ph(__A, __B),
                                              (__v32hf)_mm512_setzero_ph());
}

#define _mm512_sub_round_ph(A, B, R)                                           \
  ((__m512h)__builtin_ia32_subph512((__v32hf)(__m512h)(A),                     \
                                    (__v32hf)(__m512h)(B), (int)(R)))

#define _mm512_mask_sub_round_ph(W, U, A, B, R)                                \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_sub_round_ph((A), (B), (R)),             \
      (__v32hf)(__m512h)(W)))

#define _mm512_maskz_sub_round_ph(U, A, B, R)                                  \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_sub_round_ph((A), (B), (R)),             \
      (__v32hf)_mm512_setzero_ph()))

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_mul_ph(__m512h __A,
                                                              __m512h __B) {
  return (__m512h)((__v32hf)__A * (__v32hf)__B);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_mask_mul_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512(
      (__mmask32)__U, (__v32hf)_mm512_mul_ph(__A, __B), (__v32hf)__W);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_maskz_mul_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512((__mmask32)__U,
                                              (__v32hf)_mm512_mul_ph(__A, __B),
                                              (__v32hf)_mm512_setzero_ph());
}

#define _mm512_mul_round_ph(A, B, R)                                           \
  ((__m512h)__builtin_ia32_mulph512((__v32hf)(__m512h)(A),                     \
                                    (__v32hf)(__m512h)(B), (int)(R)))

#define _mm512_mask_mul_round_ph(W, U, A, B, R)                                \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_mul_round_ph((A), (B), (R)),             \
      (__v32hf)(__m512h)(W)))

#define _mm512_maskz_mul_round_ph(U, A, B, R)                                  \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_mul_round_ph((A), (B), (R)),             \
      (__v32hf)_mm512_setzero_ph()))

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_div_ph(__m512h __A,
                                                              __m512h __B) {
  return (__m512h)((__v32hf)__A / (__v32hf)__B);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_mask_div_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512(
      (__mmask32)__U, (__v32hf)_mm512_div_ph(__A, __B), (__v32hf)__W);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_maskz_div_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512((__mmask32)__U,
                                              (__v32hf)_mm512_div_ph(__A, __B),
                                              (__v32hf)_mm512_setzero_ph());
}

#define _mm512_div_round_ph(A, B, R)                                           \
  ((__m512h)__builtin_ia32_divph512((__v32hf)(__m512h)(A),                     \
                                    (__v32hf)(__m512h)(B), (int)(R)))

#define _mm512_mask_div_round_ph(W, U, A, B, R)                                \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_div_round_ph((A), (B), (R)),             \
      (__v32hf)(__m512h)(W)))

#define _mm512_maskz_div_round_ph(U, A, B, R)                                  \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_div_round_ph((A), (B), (R)),             \
      (__v32hf)_mm512_setzero_ph()))

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_min_ph(__m512h __A,
                                                              __m512h __B) {
  return (__m512h)__builtin_ia32_minph512((__v32hf)__A, (__v32hf)__B,
                                          _MM_FROUND_CUR_DIRECTION);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_mask_min_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512(
      (__mmask32)__U, (__v32hf)_mm512_min_ph(__A, __B), (__v32hf)__W);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_maskz_min_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512((__mmask32)__U,
                                              (__v32hf)_mm512_min_ph(__A, __B),
                                              (__v32hf)_mm512_setzero_ph());
}

#define _mm512_min_round_ph(A, B, R)                                           \
  ((__m512h)__builtin_ia32_minph512((__v32hf)(__m512h)(A),                     \
                                    (__v32hf)(__m512h)(B), (int)(R)))

#define _mm512_mask_min_round_ph(W, U, A, B, R)                                \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_min_round_ph((A), (B), (R)),             \
      (__v32hf)(__m512h)(W)))

#define _mm512_maskz_min_round_ph(U, A, B, R)                                  \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_min_round_ph((A), (B), (R)),             \
      (__v32hf)_mm512_setzero_ph()))

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_max_ph(__m512h __A,
                                                              __m512h __B) {
  return (__m512h)__builtin_ia32_maxph512((__v32hf)__A, (__v32hf)__B,
                                          _MM_FROUND_CUR_DIRECTION);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_mask_max_ph(__m512h __W, __mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512(
      (__mmask32)__U, (__v32hf)_mm512_max_ph(__A, __B), (__v32hf)__W);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_maskz_max_ph(__mmask32 __U, __m512h __A, __m512h __B) {
  return (__m512h)__builtin_ia32_selectph_512((__mmask32)__U,
                                              (__v32hf)_mm512_max_ph(__A, __B),
                                              (__v32hf)_mm512_setzero_ph());
}

#define _mm512_max_round_ph(A, B, R)                                           \
  ((__m512h)__builtin_ia32_maxph512((__v32hf)(__m512h)(A),                     \
                                    (__v32hf)(__m512h)(B), (int)(R)))

#define _mm512_mask_max_round_ph(W, U, A, B, R)                                \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_max_round_ph((A), (B), (R)),             \
      (__v32hf)(__m512h)(W)))

#define _mm512_maskz_max_round_ph(U, A, B, R)                                  \
  ((__m512h)__builtin_ia32_selectph_512(                                       \
      (__mmask32)(U), (__v32hf)_mm512_max_round_ph((A), (B), (R)),             \
      (__v32hf)_mm512_setzero_ph()))

static __inline__ __m512h __DEFAULT_FN_ATTRS512 _mm512_abs_ph(__m512h __A) {
  return (__m512h)_mm512_and_epi32(_mm512_set1_epi32(0x7FFF7FFF), (__m512i)__A);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_add_sh(__m128h __A,
                                                           __m128h __B) {
  __A[0] += __B[0];
  return __A;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_add_sh(__m128h __W,
                                                                __mmask8 __U,
                                                                __m128h __A,
                                                                __m128h __B) {
  __A = _mm_add_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, __W);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_maskz_add_sh(__mmask8 __U,
                                                                 __m128h __A,
                                                                 __m128h __B) {
  __A = _mm_add_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, _mm_setzero_ph());
}

#define _mm_add_round_sh(A, B, R)                                              \
  ((__m128h)__builtin_ia32_addsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)-1, (int)(R)))

#define _mm_mask_add_round_sh(W, U, A, B, R)                                   \
  ((__m128h)__builtin_ia32_addsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)(__m128h)(W),        \
      (__mmask8)(U), (int)(R)))

#define _mm_maskz_add_round_sh(U, A, B, R)                                     \
  ((__m128h)__builtin_ia32_addsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)(U), (int)(R)))

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_sub_sh(__m128h __A,
                                                           __m128h __B) {
  __A[0] -= __B[0];
  return __A;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_sub_sh(__m128h __W,
                                                                __mmask8 __U,
                                                                __m128h __A,
                                                                __m128h __B) {
  __A = _mm_sub_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, __W);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_maskz_sub_sh(__mmask8 __U,
                                                                 __m128h __A,
                                                                 __m128h __B) {
  __A = _mm_sub_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, _mm_setzero_ph());
}

#define _mm_sub_round_sh(A, B, R)                                              \
  ((__m128h)__builtin_ia32_subsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)-1, (int)(R)))

#define _mm_mask_sub_round_sh(W, U, A, B, R)                                   \
  ((__m128h)__builtin_ia32_subsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)(__m128h)(W),        \
      (__mmask8)(U), (int)(R)))

#define _mm_maskz_sub_round_sh(U, A, B, R)                                     \
  ((__m128h)__builtin_ia32_subsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)(U), (int)(R)))

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mul_sh(__m128h __A,
                                                           __m128h __B) {
  __A[0] *= __B[0];
  return __A;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_mul_sh(__m128h __W,
                                                                __mmask8 __U,
                                                                __m128h __A,
                                                                __m128h __B) {
  __A = _mm_mul_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, __W);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_maskz_mul_sh(__mmask8 __U,
                                                                 __m128h __A,
                                                                 __m128h __B) {
  __A = _mm_mul_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, _mm_setzero_ph());
}

#define _mm_mul_round_sh(A, B, R)                                              \
  ((__m128h)__builtin_ia32_mulsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)-1, (int)(R)))

#define _mm_mask_mul_round_sh(W, U, A, B, R)                                   \
  ((__m128h)__builtin_ia32_mulsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)(__m128h)(W),        \
      (__mmask8)(U), (int)(R)))

#define _mm_maskz_mul_round_sh(U, A, B, R)                                     \
  ((__m128h)__builtin_ia32_mulsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)(U), (int)(R)))

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_div_sh(__m128h __A,
                                                           __m128h __B) {
  __A[0] /= __B[0];
  return __A;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_div_sh(__m128h __W,
                                                                __mmask8 __U,
                                                                __m128h __A,
                                                                __m128h __B) {
  __A = _mm_div_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, __W);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_maskz_div_sh(__mmask8 __U,
                                                                 __m128h __A,
                                                                 __m128h __B) {
  __A = _mm_div_sh(__A, __B);
  return __builtin_ia32_selectsh_128(__U, __A, _mm_setzero_ph());
}

#define _mm_div_round_sh(A, B, R)                                              \
  ((__m128h)__builtin_ia32_divsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)-1, (int)(R)))

#define _mm_mask_div_round_sh(W, U, A, B, R)                                   \
  ((__m128h)__builtin_ia32_divsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)(__m128h)(W),        \
      (__mmask8)(U), (int)(R)))

#define _mm_maskz_div_round_sh(U, A, B, R)                                     \
  ((__m128h)__builtin_ia32_divsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)(U), (int)(R)))

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_min_sh(__m128h __A,
                                                           __m128h __B) {
  return (__m128h)__builtin_ia32_minsh_round_mask(
      (__v8hf)__A, (__v8hf)__B, (__v8hf)_mm_setzero_ph(), (__mmask8)-1,
      _MM_FROUND_CUR_DIRECTION);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_min_sh(__m128h __W,
                                                                __mmask8 __U,
                                                                __m128h __A,
                                                                __m128h __B) {
  return (__m128h)__builtin_ia32_minsh_round_mask((__v8hf)__A, (__v8hf)__B,
                                                  (__v8hf)__W, (__mmask8)__U,
                                                  _MM_FROUND_CUR_DIRECTION);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_maskz_min_sh(__mmask8 __U,
                                                                 __m128h __A,
                                                                 __m128h __B) {
  return (__m128h)__builtin_ia32_minsh_round_mask(
      (__v8hf)__A, (__v8hf)__B, (__v8hf)_mm_setzero_ph(), (__mmask8)__U,
      _MM_FROUND_CUR_DIRECTION);
}

#define _mm_min_round_sh(A, B, R)                                              \
  ((__m128h)__builtin_ia32_minsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)-1, (int)(R)))

#define _mm_mask_min_round_sh(W, U, A, B, R)                                   \
  ((__m128h)__builtin_ia32_minsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)(__m128h)(W),        \
      (__mmask8)(U), (int)(R)))

#define _mm_maskz_min_round_sh(U, A, B, R)                                     \
  ((__m128h)__builtin_ia32_minsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)(U), (int)(R)))

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_max_sh(__m128h __A,
                                                           __m128h __B) {
  return (__m128h)__builtin_ia32_maxsh_round_mask(
      (__v8hf)__A, (__v8hf)__B, (__v8hf)_mm_setzero_ph(), (__mmask8)-1,
      _MM_FROUND_CUR_DIRECTION);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_max_sh(__m128h __W,
                                                                __mmask8 __U,
                                                                __m128h __A,
                                                                __m128h __B) {
  return (__m128h)__builtin_ia32_maxsh_round_mask((__v8hf)__A, (__v8hf)__B,
                                                  (__v8hf)__W, (__mmask8)__U,
                                                  _MM_FROUND_CUR_DIRECTION);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_maskz_max_sh(__mmask8 __U,
                                                                 __m128h __A,
                                                                 __m128h __B) {
  return (__m128h)__builtin_ia32_maxsh_round_mask(
      (__v8hf)__A, (__v8hf)__B, (__v8hf)_mm_setzero_ph(), (__mmask8)__U,
      _MM_FROUND_CUR_DIRECTION);
}

#define _mm_max_round_sh(A, B, R)                                              \
  ((__m128h)__builtin_ia32_maxsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)-1, (int)(R)))

#define _mm_mask_max_round_sh(W, U, A, B, R)                                   \
  ((__m128h)__builtin_ia32_maxsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)(__m128h)(W),        \
      (__mmask8)(U), (int)(R)))

#define _mm_maskz_max_round_sh(U, A, B, R)                                     \
  ((__m128h)__builtin_ia32_maxsh_round_mask(                                   \
      (__v8hf)(__m128h)(A), (__v8hf)(__m128h)(B), (__v8hf)_mm_setzero_ph(),    \
      (__mmask8)(U), (int)(R)))

#define _mm512_cmp_round_ph_mask(A, B, P, R)                                   \
  ((__mmask32)__builtin_ia32_cmpph512_mask((__v32hf)(__m512h)(A),              \
                                           (__v32hf)(__m512h)(B), (int)(P),    \
                                           (__mmask32)-1, (int)(R)))

#define _mm512_mask_cmp_round_ph_mask(U, A, B, P, R)                           \
  ((__mmask32)__builtin_ia32_cmpph512_mask((__v32hf)(__m512h)(A),              \
                                           (__v32hf)(__m512h)(B), (int)(P),    \
                                           (__mmask32)(U), (int)(R)))

#define _mm512_cmp_ph_mask(A, B, P)                                            \
  _mm512_cmp_round_ph_mask((A), (B), (P), _MM_FROUND_CUR_DIRECTION)

#define _mm512_mask_cmp_ph_mask(U, A, B, P)                                    \
  _mm512_mask_cmp_round_ph_mask((U), (A), (B), (P), _MM_FROUND_CUR_DIRECTION)

#define _mm_cmp_round_sh_mask(X, Y, P, R)                                      \
  ((__mmask8)__builtin_ia32_cmpsh_mask((__v8hf)(__m128h)(X),                   \
                                       (__v8hf)(__m128h)(Y), (int)(P),         \
                                       (__mmask8)-1, (int)(R)))

#define _mm_mask_cmp_round_sh_mask(M, X, Y, P, R)                              \
  ((__mmask8)__builtin_ia32_cmpsh_mask((__v8hf)(__m128h)(X),                   \
                                       (__v8hf)(__m128h)(Y), (int)(P),         \
                                       (__mmask8)(M), (int)(R)))

#define _mm_cmp_sh_mask(X, Y, P)                                               \
  ((__mmask8)__builtin_ia32_cmpsh_mask(                                        \
      (__v8hf)(__m128h)(X), (__v8hf)(__m128h)(Y), (int)(P), (__mmask8)-1,      \
      _MM_FROUND_CUR_DIRECTION))

#define _mm_mask_cmp_sh_mask(M, X, Y, P)                                       \
  ((__mmask8)__builtin_ia32_cmpsh_mask(                                        \
      (__v8hf)(__m128h)(X), (__v8hf)(__m128h)(Y), (int)(P), (__mmask8)(M),     \
      _MM_FROUND_CUR_DIRECTION))
// loads with vmovsh:
static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_load_sh(void const *__dp) {
  struct __mm_load_sh_struct {
    _Float16 __u;
  } __attribute__((__packed__, __may_alias__));
  _Float16 __u = ((struct __mm_load_sh_struct *)__dp)->__u;
  return (__m128h){__u, 0, 0, 0, 0, 0, 0, 0};
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128
_mm_mask_load_sh(__m128h __W, __mmask8 __U, const void *__A) {
  __m128h src = (__v8hf)__builtin_shufflevector(
      (__v8hf)__W, (__v8hf)_mm_setzero_ph(), 0, 8, 8, 8, 8, 8, 8, 8);

  return (__m128h)__builtin_ia32_loadsh128_mask((__v8hf *)__A, src, __U & 1);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128
_mm_maskz_load_sh(__mmask8 __U, const void *__A) {
  return (__m128h)__builtin_ia32_loadsh128_mask(
      (__v8hf *)__A, (__v8hf)_mm_setzero_ph(), __U & 1);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_load_ph(void const *__p) {
  return *(const __m512h *)__p;
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_load_ph(void const *__p) {
  return *(const __m256h *)__p;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_load_ph(void const *__p) {
  return *(const __m128h *)__p;
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_loadu_ph(void const *__p) {
  struct __loadu_ph {
    __m512h_u __v;
  } __attribute__((__packed__, __may_alias__));
  return ((const struct __loadu_ph *)__p)->__v;
}

static __inline__ __m256h __DEFAULT_FN_ATTRS256
_mm256_loadu_ph(void const *__p) {
  struct __loadu_ph {
    __m256h_u __v;
  } __attribute__((__packed__, __may_alias__));
  return ((const struct __loadu_ph *)__p)->__v;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_loadu_ph(void const *__p) {
  struct __loadu_ph {
    __m128h_u __v;
  } __attribute__((__packed__, __may_alias__));
  return ((const struct __loadu_ph *)__p)->__v;
}

// stores with vmovsh:
static __inline__ void __DEFAULT_FN_ATTRS128 _mm_store_sh(void *__dp,
                                                          __m128h __a) {
  struct __mm_store_sh_struct {
    _Float16 __u;
  } __attribute__((__packed__, __may_alias__));
  ((struct __mm_store_sh_struct *)__dp)->__u = __a[0];
}

static __inline__ void __DEFAULT_FN_ATTRS128 _mm_mask_store_sh(void *__W,
                                                               __mmask8 __U,
                                                               __m128h __A) {
  __builtin_ia32_storesh128_mask((__v8hf *)__W, __A, __U & 1);
}

static __inline__ void __DEFAULT_FN_ATTRS512 _mm512_store_ph(void *__P,
                                                             __m512h __A) {
  *(__m512h *)__P = __A;
}

static __inline__ void __DEFAULT_FN_ATTRS256 _mm256_store_ph(void *__P,
                                                             __m256h __A) {
  *(__m256h *)__P = __A;
}

static __inline__ void __DEFAULT_FN_ATTRS128 _mm_store_ph(void *__P,
                                                          __m128h __A) {
  *(__m128h *)__P = __A;
}

static __inline__ void __DEFAULT_FN_ATTRS512 _mm512_storeu_ph(void *__P,
                                                              __m512h __A) {
  struct __storeu_ph {
    __m512h_u __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_ph *)__P)->__v = __A;
}

static __inline__ void __DEFAULT_FN_ATTRS256 _mm256_storeu_ph(void *__P,
                                                              __m256h __A) {
  struct __storeu_ph {
    __m256h_u __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_ph *)__P)->__v = __A;
}

static __inline__ void __DEFAULT_FN_ATTRS128 _mm_storeu_ph(void *__P,
                                                           __m128h __A) {
  struct __storeu_ph {
    __m128h_u __v;
  } __attribute__((__packed__, __may_alias__));
  ((struct __storeu_ph *)__P)->__v = __A;
}

// moves with vmovsh:
static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_move_sh(__m128h __a,
                                                            __m128h __b) {
  __a[0] = __b[0];
  return __a;
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_mask_move_sh(__m128h __W,
                                                                 __mmask8 __U,
                                                                 __m128h __A,
                                                                 __m128h __B) {
  return __builtin_ia32_selectsh_128(__U, _mm_move_sh(__A, __B), __W);
}

static __inline__ __m128h __DEFAULT_FN_ATTRS128 _mm_maskz_move_sh(__mmask8 __U,
                                                                  __m128h __A,
                                                                  __m128h __B) {
  return __builtin_ia32_selectsh_128(__U, _mm_move_sh(__A, __B),
                                     _mm_setzero_ph());
}

// vmovw:
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm_cvtsi16_si128(short __a) {
  return (__m128i)(__v8hi){__a, 0, 0, 0, 0, 0, 0, 0};
}

static __inline__ short __DEFAULT_FN_ATTRS128 _mm_cvtsi128_si16(__m128i __a) {
  __v8hi __b = (__v8hi)__a;
  return __b[0];
}

static __inline__ _Float16 __DEFAULT_FN_ATTRS512
_mm512_reduce_add_ph(__m512h __W) {
  return __builtin_ia32_reduce_fadd_ph512(-0.0f16, __W);
}

static __inline__ _Float16 __DEFAULT_FN_ATTRS512
_mm512_reduce_mul_ph(__m512h __W) {
  return __builtin_ia32_reduce_fmul_ph512(1.0f16, __W);
}

static __inline__ _Float16 __DEFAULT_FN_ATTRS512
_mm512_reduce_max_ph(__m512h __V) {
  return __builtin_ia32_reduce_fmax_ph512(__V);
}

static __inline__ _Float16 __DEFAULT_FN_ATTRS512
_mm512_reduce_min_ph(__m512h __V) {
  return __builtin_ia32_reduce_fmin_ph512(__V);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_mask_blend_ph(__mmask32 __U, __m512h __A, __m512h __W) {
  return (__m512h)__builtin_ia32_selectph_512((__mmask32)__U, (__v32hf)__W,
                                              (__v32hf)__A);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_permutex2var_ph(__m512h __A, __m512i __I, __m512h __B) {
  return (__m512h)__builtin_ia32_vpermi2varhi512((__v32hi)__A, (__v32hi)__I,
                                                 (__v32hi)__B);
}

static __inline__ __m512h __DEFAULT_FN_ATTRS512
_mm512_permutexvar_ph(__m512i __A, __m512h __B) {
  return (__m512h)__builtin_ia32_permvarhi512((__v32hi)__B, (__v32hi)__A);
}

#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS256
#undef __DEFAULT_FN_ATTRS512

#endif
