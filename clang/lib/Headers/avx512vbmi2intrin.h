/*===------------- avx512vbmi2intrin.h - VBMI2 intrinsics ------------------===
 *
 *
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *===-----------------------------------------------------------------------===
 */
#ifndef __IMMINTRIN_H
#error "Never use <avx512vbmi2intrin.h> directly; include <immintrin.h> instead."
#endif

#ifndef __AVX512VBMI2INTRIN_H
#define __AVX512VBMI2INTRIN_H

/* Define the default attributes for the functions in this file. */
#define __DEFAULT_FN_ATTRS                                                     \
  __attribute__((__always_inline__, __nodebug__, __target__("avx512vbmi2"),    \
                 __min_vector_width__(512)))

#if defined(__cplusplus) && (__cplusplus >= 201103L)
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS constexpr
#else
#define __DEFAULT_FN_ATTRS_CONSTEXPR __DEFAULT_FN_ATTRS
#endif

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_compress_epi16(__m512i __S, __mmask32 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_compresshi512_mask ((__v32hi) __D,
              (__v32hi) __S,
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_compress_epi16(__mmask32 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_compresshi512_mask ((__v32hi) __D,
              (__v32hi) _mm512_setzero_si512(),
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_compress_epi8(__m512i __S, __mmask64 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_compressqi512_mask ((__v64qi) __D,
              (__v64qi) __S,
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_compress_epi8(__mmask64 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_compressqi512_mask ((__v64qi) __D,
              (__v64qi) _mm512_setzero_si512(),
              __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm512_mask_compressstoreu_epi16(void *__P, __mmask32 __U, __m512i __D)
{
  __builtin_ia32_compressstorehi512_mask ((__v32hi *) __P, (__v32hi) __D,
              __U);
}

static __inline__ void __DEFAULT_FN_ATTRS
_mm512_mask_compressstoreu_epi8(void *__P, __mmask64 __U, __m512i __D)
{
  __builtin_ia32_compressstoreqi512_mask ((__v64qi *) __P, (__v64qi) __D,
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_expand_epi16(__m512i __S, __mmask32 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_expandhi512_mask ((__v32hi) __D,
              (__v32hi) __S,
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_expand_epi16(__mmask32 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_expandhi512_mask ((__v32hi) __D,
              (__v32hi) _mm512_setzero_si512(),
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_expand_epi8(__m512i __S, __mmask64 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_expandqi512_mask ((__v64qi) __D,
              (__v64qi) __S,
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_expand_epi8(__mmask64 __U, __m512i __D)
{
  return (__m512i) __builtin_ia32_expandqi512_mask ((__v64qi) __D,
              (__v64qi) _mm512_setzero_si512(),
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_expandloadu_epi16(__m512i __S, __mmask32 __U, void const *__P)
{
  return (__m512i) __builtin_ia32_expandloadhi512_mask ((const __v32hi *)__P,
              (__v32hi) __S,
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_expandloadu_epi16(__mmask32 __U, void const *__P)
{
  return (__m512i) __builtin_ia32_expandloadhi512_mask ((const __v32hi *)__P,
              (__v32hi) _mm512_setzero_si512(),
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_mask_expandloadu_epi8(__m512i __S, __mmask64 __U, void const *__P)
{
  return (__m512i) __builtin_ia32_expandloadqi512_mask ((const __v64qi *)__P,
              (__v64qi) __S,
              __U);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS
_mm512_maskz_expandloadu_epi8(__mmask64 __U, void const *__P)
{
  return (__m512i) __builtin_ia32_expandloadqi512_mask ((const __v64qi *)__P,
              (__v64qi) _mm512_setzero_si512(),
              __U);
}

#define _mm512_shldi_epi64(A, B, I) \
  ((__m512i)__builtin_ia32_vpshldq512((__v8di)(__m512i)(A), \
                                      (__v8di)(__m512i)(B), (int)(I)))

#define _mm512_mask_shldi_epi64(S, U, A, B, I) \
  ((__m512i)__builtin_ia32_selectq_512((__mmask8)(U), \
                                     (__v8di)_mm512_shldi_epi64((A), (B), (I)), \
                                     (__v8di)(__m512i)(S)))

#define _mm512_maskz_shldi_epi64(U, A, B, I) \
  ((__m512i)__builtin_ia32_selectq_512((__mmask8)(U), \
                                     (__v8di)_mm512_shldi_epi64((A), (B), (I)), \
                                     (__v8di)_mm512_setzero_si512()))

#define _mm512_shldi_epi32(A, B, I) \
  ((__m512i)__builtin_ia32_vpshldd512((__v16si)(__m512i)(A), \
                                      (__v16si)(__m512i)(B), (int)(I)))

#define _mm512_mask_shldi_epi32(S, U, A, B, I) \
  ((__m512i)__builtin_ia32_selectd_512((__mmask16)(U), \
                                    (__v16si)_mm512_shldi_epi32((A), (B), (I)), \
                                    (__v16si)(__m512i)(S)))

#define _mm512_maskz_shldi_epi32(U, A, B, I) \
  ((__m512i)__builtin_ia32_selectd_512((__mmask16)(U), \
                                    (__v16si)_mm512_shldi_epi32((A), (B), (I)), \
                                    (__v16si)_mm512_setzero_si512()))

#define _mm512_shldi_epi16(A, B, I) \
  ((__m512i)__builtin_ia32_vpshldw512((__v32hi)(__m512i)(A), \
                                      (__v32hi)(__m512i)(B), (int)(I)))

#define _mm512_mask_shldi_epi16(S, U, A, B, I) \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U), \
                                    (__v32hi)_mm512_shldi_epi16((A), (B), (I)), \
                                    (__v32hi)(__m512i)(S)))

#define _mm512_maskz_shldi_epi16(U, A, B, I) \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U), \
                                    (__v32hi)_mm512_shldi_epi16((A), (B), (I)), \
                                    (__v32hi)_mm512_setzero_si512()))

#define _mm512_shrdi_epi64(A, B, I) \
  ((__m512i)__builtin_ia32_vpshrdq512((__v8di)(__m512i)(A), \
                                      (__v8di)(__m512i)(B), (int)(I)))

#define _mm512_mask_shrdi_epi64(S, U, A, B, I) \
  ((__m512i)__builtin_ia32_selectq_512((__mmask8)(U), \
                                     (__v8di)_mm512_shrdi_epi64((A), (B), (I)), \
                                     (__v8di)(__m512i)(S)))

#define _mm512_maskz_shrdi_epi64(U, A, B, I) \
  ((__m512i)__builtin_ia32_selectq_512((__mmask8)(U), \
                                     (__v8di)_mm512_shrdi_epi64((A), (B), (I)), \
                                     (__v8di)_mm512_setzero_si512()))

#define _mm512_shrdi_epi32(A, B, I) \
  ((__m512i)__builtin_ia32_vpshrdd512((__v16si)(__m512i)(A), \
                                      (__v16si)(__m512i)(B), (int)(I)))

#define _mm512_mask_shrdi_epi32(S, U, A, B, I) \
  ((__m512i)__builtin_ia32_selectd_512((__mmask16)(U), \
                                    (__v16si)_mm512_shrdi_epi32((A), (B), (I)), \
                                    (__v16si)(__m512i)(S)))

#define _mm512_maskz_shrdi_epi32(U, A, B, I) \
  ((__m512i)__builtin_ia32_selectd_512((__mmask16)(U), \
                                    (__v16si)_mm512_shrdi_epi32((A), (B), (I)), \
                                    (__v16si)_mm512_setzero_si512()))

#define _mm512_shrdi_epi16(A, B, I) \
  ((__m512i)__builtin_ia32_vpshrdw512((__v32hi)(__m512i)(A), \
                                      (__v32hi)(__m512i)(B), (int)(I)))

#define _mm512_mask_shrdi_epi16(S, U, A, B, I) \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U), \
                                    (__v32hi)_mm512_shrdi_epi16((A), (B), (I)), \
                                    (__v32hi)(__m512i)(S)))

#define _mm512_maskz_shrdi_epi16(U, A, B, I) \
  ((__m512i)__builtin_ia32_selectw_512((__mmask32)(U), \
                                    (__v32hi)_mm512_shrdi_epi16((A), (B), (I)), \
                                    (__v32hi)_mm512_setzero_si512()))

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_shldv_epi64(__m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_elementwise_fshl((__v8du)__A, (__v8du)__B,
                                             (__v8du)__C);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_mask_shldv_epi64(__m512i __A, __mmask8 __U, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectq_512(__U,
                                      (__v8di)_mm512_shldv_epi64(__A, __B, __C),
                                      (__v8di)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_maskz_shldv_epi64(__mmask8 __U, __m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectq_512(__U,
                                      (__v8di)_mm512_shldv_epi64(__A, __B, __C),
                                      (__v8di)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_shldv_epi32(__m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_elementwise_fshl((__v16su)__A, (__v16su)__B,
                                             (__v16su)__C);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_mask_shldv_epi32(__m512i __A, __mmask16 __U, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                     (__v16si)_mm512_shldv_epi32(__A, __B, __C),
                                     (__v16si)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_maskz_shldv_epi32(__mmask16 __U, __m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectd_512(__U,
                                     (__v16si)_mm512_shldv_epi32(__A, __B, __C),
                                     (__v16si)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_shldv_epi16(__m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_elementwise_fshl((__v32hu)__A, (__v32hu)__B,
                                             (__v32hu)__C);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_mask_shldv_epi16(__m512i __A, __mmask32 __U, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectw_512(__U,
                                     (__v32hi)_mm512_shldv_epi16(__A, __B, __C),
                                     (__v32hi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_maskz_shldv_epi16(__mmask32 __U, __m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectw_512(__U,
                                     (__v32hi)_mm512_shldv_epi16(__A, __B, __C),
                                     (__v32hi)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_shrdv_epi64(__m512i __A, __m512i __B, __m512i __C)
{
  // Ops __A and __B are swapped.
  return (__m512i)__builtin_elementwise_fshr((__v8du)__B, (__v8du)__A,
                                             (__v8du)__C);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_mask_shrdv_epi64(__m512i __A, __mmask8 __U, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectq_512(__U,
                                      (__v8di)_mm512_shrdv_epi64(__A, __B, __C),
                                      (__v8di)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_maskz_shrdv_epi64(__mmask8 __U, __m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectq_512(__U,
                                      (__v8di)_mm512_shrdv_epi64(__A, __B, __C),
                                      (__v8di)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_shrdv_epi32(__m512i __A, __m512i __B, __m512i __C)
{
  // Ops __A and __B are swapped.
  return (__m512i)__builtin_elementwise_fshr((__v16su)__B, (__v16su)__A,
                                             (__v16su)__C);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_mask_shrdv_epi32(__m512i __A, __mmask16 __U, __m512i __B, __m512i __C)
{
  return (__m512i) __builtin_ia32_selectd_512(__U,
                                     (__v16si)_mm512_shrdv_epi32(__A, __B, __C),
                                     (__v16si)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_maskz_shrdv_epi32(__mmask16 __U, __m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i) __builtin_ia32_selectd_512(__U,
                                     (__v16si)_mm512_shrdv_epi32(__A, __B, __C),
                                     (__v16si)_mm512_setzero_si512());
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_shrdv_epi16(__m512i __A, __m512i __B, __m512i __C)
{
  // Ops __A and __B are swapped.
  return (__m512i)__builtin_elementwise_fshr((__v32hu)__B, (__v32hu)__A,
                                             (__v32hu)__C);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_mask_shrdv_epi16(__m512i __A, __mmask32 __U, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectw_512(__U,
                                     (__v32hi)_mm512_shrdv_epi16(__A, __B, __C),
                                     (__v32hi)__A);
}

static __inline__ __m512i __DEFAULT_FN_ATTRS_CONSTEXPR
_mm512_maskz_shrdv_epi16(__mmask32 __U, __m512i __A, __m512i __B, __m512i __C)
{
  return (__m512i)__builtin_ia32_selectw_512(__U,
                                     (__v32hi)_mm512_shrdv_epi16(__A, __B, __C),
                                     (__v32hi)_mm512_setzero_si512());
}


#undef __DEFAULT_FN_ATTRS
#undef __DEFAULT_FN_ATTRS_CONSTEXPR

#endif

