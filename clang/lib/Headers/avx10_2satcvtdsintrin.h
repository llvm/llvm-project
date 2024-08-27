/*===----------- avx10_2satcvtdsintrin.h - AVX512SATCVTDS intrinsics --------===
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __IMMINTRIN_H
#error                                                                         \
    "Never use <avx10_2satcvtdsintrin.h> directly; include <immintrin.h> instead."
#endif // __IMMINTRIN_H

#ifndef __AVX10_2SATCVTDSINTRIN_H
#define __AVX10_2SATCVTDSINTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10.2-256"),    \
                 __min_vector_width__(256)))

#define __DEFAULT_FN_ATTRS128                                                  \
  __attribute__((__always_inline__, __nodebug__, __target__("avx10.2-256"),    \
                 __min_vector_width__(128)))

#define _mm_cvtts_roundsd_i32(A, R)                                            \
  ((int)__builtin_ia32_vcvttssd2si32((__v2df)(__m128)(A), (const int)(R)))

#define _mm_cvtts_roundsd_si32(A, R)                                           \
  ((int)__builtin_ia32_vcvttssd2si32((__v2df)(__m128d)(A), (const int)(R)))
                                           (const int)(R)))

#define _mm_cvtts_roundsd_u32(A, R)   \
                                             ((unsigned int)                     \
                                                  __builtin_ia32_vcvttssd2usi32( \
                                                      (__v2df)(__m128d)(A),      \
                                                      (const int)(R)))

#define _mm_cvtts_roundss_i32(A, R)  \
                                             ((int)                             \
                                                  __builtin_ia32_vcvttsss2si32( \
                                                      (__v4sf)(__m128)(A),      \
                                                      (const int)(R)))

#define _mm_cvtts_roundss_si32(A,    \
                                                                          R)    \
                                             ((int)                             \
                                                  __builtin_ia32_vcvttsss2si32( \
                                                      (__v4sf)(__m128)(A),      \
                                                      (const int)(R)))

#define _mm_cvtts_roundss_u32(A, R)   \
                                             ((unsigned int)                     \
                                                  __builtin_ia32_vcvttsss2usi32( \
                                                      (__v4sf)(__m128)(A),       \
                                                      (const int)(R)))

#ifdef __x86_64__
#define _mm_cvtts_roundss_u64(A, R)   \
                                             ((unsigned long long)               \
                                                  __builtin_ia32_vcvttsss2usi64( \
                                                      (__v4sf)(__m128)(A),       \
                                                      (const int)(R)))

#define _mm_cvtts_roundsd_u64(A, R)   \
                                             ((unsigned long long)               \
                                                  __builtin_ia32_vcvttssd2usi64( \
                                                      (__v2df)(__m128d)(A),      \
                                                      (const int)(R)))

#define _mm_cvtts_roundss_i64(A, R)  \
                                             ((long long)                       \
                                                  __builtin_ia32_vcvttsss2si64( \
                                                      (__v4sf)(__m128)(A),      \
                                                      (const int)(R)))

#define _mm_cvtts_roundss_si64(A,    \
                                                                          R)    \
                                             ((long long)                       \
                                                  __builtin_ia32_vcvttsss2si64( \
                                                      (__v4sf)(__m128)(A),      \
                                                      (const int)(R)))

#define _mm_cvtts_roundsd_si64(A,    \
                                                                          R)    \
                                             ((long long)                       \
                                                  __builtin_ia32_vcvttssd2si64( \
                                                      (__v2df)(__m128d)(A),     \
                                                      (const int)(R)))

#define _mm_cvtts_roundsd_i64(A, R) \
  ((long long)__builtin_ia32_vcvttssd2si64((__v2df)(__m128d)(A),               \
#endif /* __x86_64__ */

                                           //  128 Bit : Double -> int
#define _mm_cvttspd_epi32(A)               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttpd2dqs128_mask( \
                                                      (__v2df)(__m128d)A,             \
                                                      (__v4si)(__m128i)               \
                                                          _mm_undefined_si128(),      \
                                                      (__mmask8)(-1)))

#define _mm_mask_cvttspd_epi32(            \
                                               W, U, A)                               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttpd2dqs128_mask( \
                                                      (__v2df)(__m128d)A,             \
                                                      (__v4si)(__m128i)W,             \
                                                      (__mmask8)U))

#define _mm_maskz_cvttspd_epi32(U,         \
                                                                           A)         \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttpd2dqs128_mask( \
                                                      (__v2df)(__m128d)A,             \
                                                      (__v4si)(__m128i)               \
                                                          _mm_setzero_si128(),        \
                                                      (__mmask8)U))

//  256 Bit : Double -> int
static __inline__ __m128i __DEFAULT_FN_ATTRS128 _mm256_cvttspd_epi32(__m256d A) {
                                             return (
                                                 (__m128i)__builtin_ia32_vcvttpd2dqs256_round_mask(
                                                     (__v4df)(__m256d)A,
                                                     (__v4si)
                                                         _mm_undefined_si128(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m128i
                                               __DEFAULT_FN_ATTRS128
                                               _mm256_mask_cvttspd_epi32(
                                                   __m128i W, __mmask8 U,
                                                   __m256d A) {
                                             return (
                                                 (__m128i)__builtin_ia32_vcvttpd2dqs256_round_mask(
                                                     (__v4df)A, (__v4si)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m128i
                                               __DEFAULT_FN_ATTRS128
                                               _mm256_maskz_cvttspd_epi32(
                                                   __mmask8 U, __m256d A) {
                                             return (
                                                 (__m128i)__builtin_ia32_vcvttpd2dqs256_round_mask(
                                                     (__v4df)A,
                                                     (__v4si)
                                                         _mm_setzero_si128(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundpd_epi32(              \
                                               A, R)                                        \
                                             ((__m128i)                                     \
                                                  __builtin_ia32_vcvttpd2dqs256_round_mask( \
                                                      (__v4df)(__m256d)A,                   \
                                                      (__v4si)(__m128i)                     \
                                                          _mm_undefined_si128(),            \
                                                      (__mmask8) - 1,                       \
                                                      (int)(R)))

#define _mm256_mask_cvtts_roundpd_epi32(         \
                                               W, U, A, R)                                  \
                                             ((__m128i)                                     \
                                                  __builtin_ia32_vcvttpd2dqs256_round_mask( \
                                                      (__v4df)(__m256d)A,                   \
                                                      (__v4si)(__m128i)W,                   \
                                                      (__mmask8)U, (int)(R)))

#define _mm256_maskz_cvtts_roundpd_epi32(        \
                                               U, A, R)                                     \
                                             ((__m128i)                                     \
                                                  __builtin_ia32_vcvttpd2dqs256_round_mask( \
                                                      (__v4df)(__m256d)A,                   \
                                                      (__v4si)(__m128i)                     \
                                                          _mm_setzero_si128(),              \
                                                      (__mmask8)U, (int)(R)))

                                           //  128 Bit : Double -> uint
#define _mm_cvttspd_epu32(A)                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttpd2udqs128_mask( \
                                                      (__v2df)(__m128d)A,              \
                                                      (__v4si)(__m128i)                \
                                                          _mm_undefined_si128(),       \
                                                      (__mmask8)(-1)))

#define _mm_mask_cvttspd_epu32(             \
                                               W, U, A)                                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttpd2udqs128_mask( \
                                                      ((__v2df)(__m128d)A),            \
                                                      (__v4si)(__m128i)W,              \
                                                      (__mmask8)U))

#define _mm_maskz_cvttspd_epu32(U,          \
                                                                           A)          \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttpd2udqs128_mask( \
                                                      (__v2df)(__m128d)A,              \
                                                      (__v4si)(__m128i)                \
                                                          _mm_setzero_si128(),         \
                                                      (__mmask8)U))

                                           //  256 Bit : Double -> uint
                                           static __inline__ __m128i
                                               __DEFAULT_FN_ATTRS128
                                               _mm256_cvttspd_epu32(__m256d A) {
                                             return (
                                                 (__m128i)__builtin_ia32_vcvttpd2udqs256_round_mask(
                                                     (__v4df)A,
                                                     (__v4si)
                                                         _mm_undefined_si128(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m128i
                                               __DEFAULT_FN_ATTRS128
                                               _mm256_mask_cvttspd_epu32(
                                                   __m128i W, __mmask8 U,
                                                   __m256d A) {
                                             return (
                                                 (__m128i)__builtin_ia32_vcvttpd2udqs256_round_mask(
                                                     (__v4df)A, (__v4si)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m128i
                                               __DEFAULT_FN_ATTRS128
                                               _mm256_maskz_cvttspd_epu32(
                                                   __mmask8 U, __m256d A) {
                                             return (
                                                 (__m128i)__builtin_ia32_vcvttpd2udqs256_round_mask(
                                                     (__v4df)A,
                                                     (__v4si)
                                                         _mm_setzero_si128(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundpd_epu32(               \
                                               A, R)                                         \
                                             ((__m128i)                                      \
                                                  __builtin_ia32_vcvttpd2udqs256_round_mask( \
                                                      (__v4df)(__m256d)A,                    \
                                                      (__v4si)(__m128i)                      \
                                                          _mm_undefined_si128(),             \
                                                      (__mmask8) - 1,                        \
                                                      (int)(R)))

#define _mm256_mask_cvtts_roundpd_epu32(          \
                                               W, U, A, R)                                   \
                                             ((__m128i)                                      \
                                                  __builtin_ia32_vcvttpd2udqs256_round_mask( \
                                                      (__v4df)(__m256d)A,                    \
                                                      (__v4si)(__m128i)W,                    \
                                                      (__mmask8)U, (int)(R)))

#define _mm256_maskz_cvtts_roundpd_epu32(         \
                                               U, A, R)                                      \
                                             ((__m128i)                                      \
                                                  __builtin_ia32_vcvttpd2udqs256_round_mask( \
                                                      (__v4df)(__m256d)A,                    \
                                                      (__v4si)(__m128i)                      \
                                                          _mm_setzero_si128(),               \
                                                      (__mmask8)U, (int)(R)))

                                           //  128 Bit : Double -> long
#define _mm_cvttspd_epi64(A)               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttpd2qqs128_mask( \
                                                      (__v2df)(__m128d)A,             \
                                                      (__v2di)                        \
                                                          _mm_undefined_si128(),      \
                                                      (__mmask8) - 1))

#define _mm_mask_cvttspd_epi64(            \
                                               W, U, A)                               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttpd2qqs128_mask( \
                                                      (__v2df)(__m128d)A,             \
                                                      (__v2di)W, (__mmask8)U))

#define _mm_maskz_cvttspd_epi64(U,         \
                                                                           A)         \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttpd2qqs128_mask( \
                                                      (__v2df)(__m128d)A,             \
                                                      (__v2di)                        \
                                                          _mm_setzero_si128(),        \
                                                      (__mmask8)U))

                                           //  256 Bit : Double -> long
                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_cvttspd_epi64(__m256d A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttpd2qqs256_round_mask(
                                                     (__v4df)A,
                                                     (__v4di)
                                                         _mm256_undefined_si256(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_mask_cvttspd_epi64(
                                                   __m256i W, __mmask8 U,
                                                   __m256d A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttpd2qqs256_round_mask(
                                                     (__v4df)A, (__v4di)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_maskz_cvttspd_epi64(
                                                   __mmask8 U, __m256d A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttpd2qqs256_round_mask(
                                                     (__v4df)A,
                                                     (__v4di)
                                                         _mm256_setzero_si256(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundpd_epi64(                   \
                                               A, R)                                             \
                                             ((__m256i)__builtin_ia32_vcvttpd2qqs256_round_mask( \
                                                 (__v4df)A,                                      \
                                                 (__v4di)                                        \
                                                     _mm256_undefined_si256(),                   \
                                                 (__mmask8) - 1, (int)R))

#define _mm256_mask_cvtts_roundpd_epi64(         \
                                               W, U, A, R)                                  \
                                             ((__m256i)                                     \
                                                  __builtin_ia32_vcvttpd2qqs256_round_mask( \
                                                      (__v4df)A, (__v4di)W,                 \
                                                      (__mmask8)U, (int)R))

#define _mm256_maskz_cvtts_roundpd_epi64(        \
                                               U, A, R)                                     \
                                             ((__m256i)                                     \
                                                  __builtin_ia32_vcvttpd2qqs256_round_mask( \
                                                      (__v4df)A,                            \
                                                      (__v4di)                              \
                                                          _mm256_setzero_si256(),           \
                                                      (__mmask8)U, (int)R))

                                           //  128 Bit : Double -> ulong
#define _mm_cvttspd_epu64(A)                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttpd2uqqs128_mask( \
                                                      (__v2df)(__m128d)A,              \
                                                      (__v2di)                         \
                                                          _mm_undefined_si128(),       \
                                                      (__mmask8) - 1))

#define _mm_mask_cvttspd_epu64(             \
                                               W, U, A)                                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttpd2uqqs128_mask( \
                                                      (__v2df)(__m128d)A,              \
                                                      (__v2di)W, (__mmask8)U))

#define _mm_maskz_cvttspd_epu64(U,          \
                                                                           A)          \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttpd2uqqs128_mask( \
                                                      (__v2df)(__m128d)A,              \
                                                      (__v2di)                         \
                                                          _mm_setzero_si128(),         \
                                                      (__mmask8)U))

                                           //  256 Bit : Double -> ulong

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_cvttspd_epu64(__m256d A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttpd2uqqs256_round_mask(
                                                     (__v4df)A,
                                                     (__v4di)
                                                         _mm256_undefined_si256(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_mask_cvttspd_epu64(
                                                   __m256i W, __mmask8 U,
                                                   __m256d A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttpd2uqqs256_round_mask(
                                                     (__v4df)A, (__v4di)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_maskz_cvttspd_epu64(
                                                   __mmask8 U, __m256d A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttpd2uqqs256_round_mask(
                                                     (__v4df)A,
                                                     (__v4di)
                                                         _mm256_setzero_si256(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundpd_epu64(                    \
                                               A, R)                                              \
                                             ((__m256i)__builtin_ia32_vcvttpd2uqqs256_round_mask( \
                                                 (__v4df)A,                                       \
                                                 (__v4di)                                         \
                                                     _mm256_undefined_si256(),                    \
                                                 (__mmask8) - 1, (int)R))

#define _mm256_mask_cvtts_roundpd_epu64(          \
                                               W, U, A, R)                                   \
                                             ((__m256i)                                      \
                                                  __builtin_ia32_vcvttpd2uqqs256_round_mask( \
                                                      (__v4df)A, (__v4di)W,                  \
                                                      (__mmask8)U, (int)R))

#define _mm256_maskz_cvtts_roundpd_epu64(         \
                                               U, A, R)                                      \
                                             ((__m256i)                                      \
                                                  __builtin_ia32_vcvttpd2uqqs256_round_mask( \
                                                      (__v4df)A,                             \
                                                      (__v4di)                               \
                                                          _mm256_setzero_si256(),            \
                                                      (__mmask8)U, (int)R))

                                           //  128 Bit : float -> int
#define _mm_cvttsps_epi32(A)               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttps2dqs128_mask( \
                                                      (__v4sf)(__m128)A,              \
                                                      (__v4si)(__m128i)               \
                                                          _mm_undefined_si128(),      \
                                                      (__mmask8)(-1)))

#define _mm_mask_cvttsps_epi32(            \
                                               W, U, A)                               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttps2dqs128_mask( \
                                                      (__v4sf)(__m128)A,              \
                                                      (__v4si)(__m128i)W,             \
                                                      (__mmask8)U))

#define _mm_maskz_cvttsps_epi32(U,         \
                                                                           A)         \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttps2dqs128_mask( \
                                                      (__v4sf)(__m128)A,              \
                                                      (__v4si)(__m128i)               \
                                                          _mm_setzero_si128(),        \
                                                      (__mmask8)U))

                                           //  256 Bit : float -> int
                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_cvttsps_epi32(__m256 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2dqs256_round_mask(
                                                     (__v8sf)A,
                                                     (__v8si)
                                                         _mm256_undefined_si256(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_mask_cvttsps_epi32(
                                                   __m256i W, __mmask8 U,
                                                   __m256 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2dqs256_round_mask(
                                                     (__v8sf)(__m256)A,
                                                     (__v8si)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_maskz_cvttsps_epi32(
                                                   __mmask8 U, __m256 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2dqs256_round_mask(
                                                     (__v8sf)(__m256)A,
                                                     (__v8si)
                                                         _mm256_setzero_si256(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundps_epi32(                   \
                                               A, R)                                             \
                                             ((__m256i)__builtin_ia32_vcvttps2dqs256_round_mask( \
                                                 (__v8sf)(__m256)A,                              \
                                                 (__v8si)(__m256i)                               \
                                                     _mm256_undefined_si256(),                   \
                                                 (__mmask8) - 1, (int)(R)))

#define _mm256_mask_cvtts_roundps_epi32(         \
                                               W, U, A, R)                                  \
                                             ((__m256i)                                     \
                                                  __builtin_ia32_vcvttps2dqs256_round_mask( \
                                                      (__v8sf)(__m256)A,                    \
                                                      (__v8si)(__m256i)W,                   \
                                                      (__mmask8)U, (int)(R)))

#define _mm256_maskz_cvtts_roundps_epi32(        \
                                               U, A, R)                                     \
                                             ((__m256i)                                     \
                                                  __builtin_ia32_vcvttps2dqs256_round_mask( \
                                                      (__v8sf)(__m256)A,                    \
                                                      (__v8si)(__m256i)                     \
                                                          _mm256_setzero_si256(),           \
                                                      (__mmask8)U, (int)(R)))

                                           //  128 Bit : float -> uint
#define _mm_cvttsps_epu32(A)                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttps2udqs128_mask( \
                                                      (__v4sf)(__m128)A,               \
                                                      (__v4si)(__m128i)                \
                                                          _mm_undefined_si128(),       \
                                                      (__mmask8)(-1)))

#define _mm_mask_cvttsps_epu32(             \
                                               W, U, A)                                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttps2udqs128_mask( \
                                                      (__v4sf)(__m128)A,               \
                                                      (__v4si)(__m128i)W,              \
                                                      (__mmask8)U))

#define _mm_maskz_cvttsps_epu32(U,          \
                                                                           A)          \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttps2udqs128_mask( \
                                                      (__v4sf)(__m128)A,               \
                                                      (__v4si)(__m128i)                \
                                                          _mm_setzero_si128(),         \
                                                      (__mmask8)U))

                                           //  256 Bit : float -> uint

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_cvttsps_epu32(__m256 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2udqs256_round_mask(
                                                     (__v8sf)A,
                                                     (__v8si)
                                                         _mm256_undefined_si256(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_mask_cvttsps_epu32(
                                                   __m256i W, __mmask8 U,
                                                   __m256 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2udqs256_round_mask(
                                                     (__v8sf)A, (__v8si)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_maskz_cvttsps_epu32(
                                                   __mmask8 U, __m256 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2udqs256_round_mask(
                                                     (__v8sf)A,
                                                     (__v8si)
                                                         _mm256_setzero_si256(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundps_epu32(                    \
                                               A, R)                                              \
                                             ((__m256i)__builtin_ia32_vcvttps2udqs256_round_mask( \
                                                 (__v8sf)(__m256)A,                               \
                                                 (__v8si)(__m256i)                                \
                                                     _mm256_undefined_si256(),                    \
                                                 (__mmask8) - 1, (int)(R)))

#define _mm256_mask_cvtts_roundps_epu32(          \
                                               W, U, A, R)                                   \
                                             ((__m256i)                                      \
                                                  __builtin_ia32_vcvttps2udqs256_round_mask( \
                                                      (__v8sf)(__m256)A,                     \
                                                      (__v8si)(__m256i)W,                    \
                                                      (__mmask8)U, (int)(R)))

#define _mm256_maskz_cvtts_roundps_epu32(         \
                                               U, A, R)                                      \
                                             ((__m256i)                                      \
                                                  __builtin_ia32_vcvttps2udqs256_round_mask( \
                                                      (__v8sf)(__m256)A,                     \
                                                      (__v8si)(__m256i)                      \
                                                          _mm256_setzero_si256(),            \
                                                      (__mmask8)U, (int)(R)))

                                           // 128 bit : float -> long
#define _mm_cvttsps_epi64(A)               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttps2qqs128_mask( \
                                                      (__v4sf)(__m128)A,              \
                                                      (__v2di)                        \
                                                          _mm_undefined_si128(),      \
                                                      (__mmask8) - 1))

#define _mm_mask_cvttsps_epi64(            \
                                               W, U, A)                               \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttps2qqs128_mask( \
                                                      (__v4sf)(__m128)A,              \
                                                      (__v2di)(__m128i)W,             \
                                                      (__mmask8)U))

#define _mm_maskz_cvttsps_epi64(U,         \
                                                                           A)         \
                                             ((__m128i)                               \
                                                  __builtin_ia32_vcvttps2qqs128_mask( \
                                                      (__v4sf)(__m128)A,              \
                                                      (__v2di)                        \
                                                          _mm_setzero_si128(),        \
                                                      (__mmask8)U))
                                           /*
                                           // 256 bit : float -> long
                                           */

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_cvttsps_epi64(__m128 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2qqs256_round_mask(
                                                     (__v4sf)A,
                                                     (__v4di)
                                                         _mm256_undefined_si256(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }
                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_mask_cvttsps_epi64(
                                                   __m256i W, __mmask8 U,
                                                   __m128 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2qqs256_round_mask(
                                                     (__v4sf)A, (__v4di)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_maskz_cvttsps_epi64(
                                                   __mmask8 U, __m128 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2qqs256_round_mask(
                                                     (__v4sf)A,
                                                     (__v4di)
                                                         _mm256_setzero_si256(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundps_epi64(                   \
                                               A, R)                                             \
                                             ((__m256i)__builtin_ia32_vcvttps2qqs256_round_mask( \
                                                 (__v4sf)(__m128)A,                              \
                                                 (__v4di)                                        \
                                                     _mm256_undefined_si256(),                   \
                                                 (__mmask8) - 1, (int)R))

#define _mm256_mask_cvtts_roundps_epi64(         \
                                               W, U, A, R)                                  \
                                             ((__m256i)                                     \
                                                  __builtin_ia32_vcvttps2qqs256_round_mask( \
                                                      (__v4sf)(__m128)A,                    \
                                                      (__v4di)W, (__mmask8)U,               \
                                                      (int)R))

#define _mm256_maskz_cvtts_roundps_epi64(        \
                                               U, A, R)                                     \
                                             ((__m256i)                                     \
                                                  __builtin_ia32_vcvttps2qqs256_round_mask( \
                                                      (__v4sf)(__m128)A,                    \
                                                      (__v4di)                              \
                                                          _mm256_setzero_si256(),           \
                                                      (__mmask8)U, (int)R))

                                           // 128 bit : float -> ulong
#define _mm_cvttsps_epu64(A)                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttps2uqqs128_mask( \
                                                      (__v4sf)(__m128)A,               \
                                                      (__v2di)                         \
                                                          _mm_undefined_si128(),       \
                                                      (__mmask8) - 1))

#define _mm_mask_cvttsps_epu64(             \
                                               W, U, A)                                \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttps2uqqs128_mask( \
                                                      (__v4sf)(__m128)A,               \
                                                      (__v2di)(__m128i)W,              \
                                                      (__mmask8)U))

#define _mm_maskz_cvttsps_epu64(U,          \
                                                                           A)          \
                                             ((__m128i)                                \
                                                  __builtin_ia32_vcvttps2uqqs128_mask( \
                                                      (__v4sf)(__m128)A,               \
                                                      (__v2di)                         \
                                                          _mm_setzero_si128(),         \
                                                      (__mmask8)U))
                                           /*
                                           // 256 bit : float -> ulong
                                           */

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_cvttsps_epu64(__m128 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2uqqs256_round_mask(
                                                     (__v4sf)A,
                                                     (__v4di)
                                                         _mm256_undefined_si256(),
                                                     (__mmask8)-1,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_mask_cvttsps_epu64(
                                                   __m256i W, __mmask8 U,
                                                   __m128 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2uqqs256_round_mask(
                                                     (__v4sf)A, (__v4di)W, U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

                                           static __inline__ __m256i
                                               __DEFAULT_FN_ATTRS
                                               _mm256_maskz_cvttsps_epu64(
                                                   __mmask8 U, __m128 A) {
                                             return (
                                                 (__m256i)__builtin_ia32_vcvttps2uqqs256_round_mask(
                                                     (__v4sf)A,
                                                     (__v4di)
                                                         _mm256_setzero_si256(),
                                                     U,
                                                     _MM_FROUND_CUR_DIRECTION));
                                           }

#define _mm256_cvtts_roundps_epu64(                    \
                                               A, R)                                              \
                                             ((__m256i)__builtin_ia32_vcvttps2uqqs256_round_mask( \
                                                 (__v4sf)(__m128)A,                               \
                                                 (__v4di)                                         \
                                                     _mm256_undefined_si256(),                    \
                                                 (__mmask8) - 1, (int)R))

#define _mm256_mask_cvtts_roundps_epu64(          \
                                               W, U, A, R)                                   \
                                             ((__m256i)                                      \
                                                  __builtin_ia32_vcvttps2uqqs256_round_mask( \
                                                      (__v4sf)(__m128)A,                     \
                                                      (__v4di)W, (__mmask8)U,                \
                                                      (int)R))

#define _mm256_maskz_cvtts_roundps_epu64(         \
                                               U, A, R)                                      \
                                             ((__m256i)                                      \
                                                  __builtin_ia32_vcvttps2uqqs256_round_mask( \
                                                      (__v4sf)(__m128)A,                     \
                                                      (__v4di)                               \
                                                          _mm256_setzero_si256(),            \
                                                      (__mmask8)U, (int)R))

#undef __DEFAULT_FN_ATTRS128
#undef __DEFAULT_FN_ATTRS
#endif /*__AVX10_2SATCVTDSINTRIN_H*/
