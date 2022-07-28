//===-- Elementary operations for x86 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_X86_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_X86_H

#if defined(LLVM_LIBC_ARCH_X86)
#include "src/__support/CPP/type_traits.h" // ConditionalType, enable_if_t
#include "src/string/memory_utils/backend_scalar.h"

#ifdef __SSE2__
#include <immintrin.h>
#endif //  __SSE2__

#if defined(__SSE2__)
#define HAS_M128 true
#else
#define HAS_M128 false
#endif

#if defined(__AVX2__)
#define HAS_M256 true
#else
#define HAS_M256 false
#endif

#if defined(__AVX512F__) and defined(__AVX512BW__)
#define HAS_M512 true
#else
#define HAS_M512 false
#endif

namespace __llvm_libc {
struct X86Backend : public Scalar64BitBackend {
  static constexpr bool IS_BACKEND_TYPE = true;

  // Scalar types use base class implementations.
  template <typename T, Temporality TS, Aligned AS,
            cpp::enable_if_t<Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline T load(const T *src) {
    return Scalar64BitBackend::template load<T, TS, AS>(src);
  }

  // Scalar types use base class implementations.
  template <typename T, Temporality TS, Aligned AS,
            cpp::enable_if_t<Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline void store(T *dst, T value) {
    Scalar64BitBackend::template store<T, TS, AS>(dst, value);
  }

  // Scalar types use base class implementations.
  template <typename T,
            cpp::enable_if_t<Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline uint64_t notEquals(T v1, T v2) {
    return Scalar64BitBackend::template notEquals<T>(v1, v2);
  }

  // Scalar types use base class implementations.
  template <typename T,
            cpp::enable_if_t<Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline T splat(ubyte value) {
    return Scalar64BitBackend::template splat<T>(value);
  }

  // Scalar types use base class implementations.
  template <typename T,
            cpp::enable_if_t<Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline int32_t threeWayCmp(T v1, T v2) {
    return Scalar64BitBackend::template threeWayCmp<T>(v1, v2);
  }

  // X86 types are specialized below.
  template <typename T, Temporality TS, Aligned AS,
            cpp::enable_if_t<!Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline T load(const T *src);

  // X86 types are specialized below.
  template <typename T, Temporality TS, Aligned AS,
            cpp::enable_if_t<!Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline void store(T *dst, T value);

  // X86 types are specialized below.
  template <typename T,
            cpp::enable_if_t<!Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline T splat(ubyte value);

  // X86 types are specialized below.
  template <typename T,
            cpp::enable_if_t<!Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline uint64_t notEquals(T v1, T v2);

  template <typename T,
            cpp::enable_if_t<!Scalar64BitBackend::IsScalarType<T>, bool> = true>
  static inline int32_t threeWayCmp(T v1, T v2) {
    return char_diff(reinterpret_cast<char *>(&v1),
                     reinterpret_cast<char *>(&v2), notEquals(v1, v2));
  }

  // Returns the type to use to consume Size bytes.
  template <size_t Size>
  using getNextType = cpp::conditional_t<
      (HAS_M512 && Size >= 64), __m512i,
      cpp::conditional_t<
          (HAS_M256 && Size >= 32), __m256i,
          cpp::conditional_t<(HAS_M128 && Size >= 16), __m128i,
                             Scalar64BitBackend::getNextType<Size>>>>;

private:
  static inline int32_t char_diff(const char *a, const char *b, uint64_t mask) {
    const size_t diff_index = mask == 0 ? 0 : __builtin_ctzll(mask);
    const int16_t ca = (unsigned char)a[diff_index];
    const int16_t cb = (unsigned char)b[diff_index];
    return ca - cb;
  }
};

static inline void repmovsb(void *dst, const void *src, size_t runtime_size) {
  asm volatile("rep movsb"
               : "+D"(dst), "+S"(src), "+c"(runtime_size)
               :
               : "memory");
}

#define SPECIALIZE_LOAD(T, OS, AS, INTRISIC)                                   \
  template <> inline T X86Backend::load<T, OS, AS>(const T *src) {             \
    return INTRISIC(const_cast<T *>(src));                                     \
  }
#define SPECIALIZE_STORE(T, OS, AS, INTRISIC)                                  \
  template <> inline void X86Backend::store<T, OS, AS>(T * dst, T value) {     \
    INTRISIC(dst, value);                                                      \
  }

#if HAS_M128
SPECIALIZE_LOAD(__m128i, Temporality::TEMPORAL, Aligned::YES, _mm_load_si128)
SPECIALIZE_LOAD(__m128i, Temporality::TEMPORAL, Aligned::NO, _mm_loadu_si128)
SPECIALIZE_LOAD(__m128i, Temporality::NON_TEMPORAL, Aligned::YES,
                _mm_stream_load_si128)
// X86 non-temporal load needs aligned access
SPECIALIZE_STORE(__m128i, Temporality::TEMPORAL, Aligned::YES, _mm_store_si128)
SPECIALIZE_STORE(__m128i, Temporality::TEMPORAL, Aligned::NO, _mm_storeu_si128)
SPECIALIZE_STORE(__m128i, Temporality::NON_TEMPORAL, Aligned::YES,
                 _mm_stream_si128)
// X86 non-temporal store needs aligned access
template <> inline __m128i X86Backend::splat<__m128i>(ubyte value) {
  return _mm_set1_epi8(__builtin_bit_cast(char, value));
}
template <>
inline uint64_t X86Backend::notEquals<__m128i>(__m128i a, __m128i b) {
  using T = char __attribute__((__vector_size__(16)));
  return _mm_movemask_epi8(T(a) != T(b));
}
#endif // HAS_M128

#if HAS_M256
SPECIALIZE_LOAD(__m256i, Temporality::TEMPORAL, Aligned::YES, _mm256_load_si256)
SPECIALIZE_LOAD(__m256i, Temporality::TEMPORAL, Aligned::NO, _mm256_loadu_si256)
SPECIALIZE_LOAD(__m256i, Temporality::NON_TEMPORAL, Aligned::YES,
                _mm256_stream_load_si256)
// X86 non-temporal load needs aligned access
SPECIALIZE_STORE(__m256i, Temporality::TEMPORAL, Aligned::YES,
                 _mm256_store_si256)
SPECIALIZE_STORE(__m256i, Temporality::TEMPORAL, Aligned::NO,
                 _mm256_storeu_si256)
SPECIALIZE_STORE(__m256i, Temporality::NON_TEMPORAL, Aligned::YES,
                 _mm256_stream_si256)
// X86 non-temporal store needs aligned access
template <> inline __m256i X86Backend::splat<__m256i>(ubyte value) {
  return _mm256_set1_epi8(__builtin_bit_cast(char, value));
}
template <>
inline uint64_t X86Backend::notEquals<__m256i>(__m256i a, __m256i b) {
  using T = char __attribute__((__vector_size__(32)));
  return _mm256_movemask_epi8(T(a) != T(b));
}
#endif // HAS_M256

#if HAS_M512
SPECIALIZE_LOAD(__m512i, Temporality::TEMPORAL, Aligned::YES, _mm512_load_si512)
SPECIALIZE_LOAD(__m512i, Temporality::TEMPORAL, Aligned::NO, _mm512_loadu_si512)
SPECIALIZE_LOAD(__m512i, Temporality::NON_TEMPORAL, Aligned::YES,
                _mm512_stream_load_si512)
// X86 non-temporal load needs aligned access
SPECIALIZE_STORE(__m512i, Temporality::TEMPORAL, Aligned::YES,
                 _mm512_store_si512)
SPECIALIZE_STORE(__m512i, Temporality::TEMPORAL, Aligned::NO,
                 _mm512_storeu_si512)
SPECIALIZE_STORE(__m512i, Temporality::NON_TEMPORAL, Aligned::YES,
                 _mm512_stream_si512)
// X86 non-temporal store needs aligned access
template <> inline __m512i X86Backend::splat<__m512i>(ubyte value) {
  return _mm512_broadcastb_epi8(_mm_set1_epi8(__builtin_bit_cast(char, value)));
}
template <>
inline uint64_t X86Backend::notEquals<__m512i>(__m512i a, __m512i b) {
  return _mm512_cmpneq_epi8_mask(a, b);
}
#endif // HAS_M512

namespace x86 {
using _1 = SizedOp<X86Backend, 1>;
using _2 = SizedOp<X86Backend, 2>;
using _3 = SizedOp<X86Backend, 3>;
using _4 = SizedOp<X86Backend, 4>;
using _8 = SizedOp<X86Backend, 8>;
using _16 = SizedOp<X86Backend, 16>;
using _32 = SizedOp<X86Backend, 32>;
using _64 = SizedOp<X86Backend, 64>;
using _128 = SizedOp<X86Backend, 128>;
} // namespace x86

} // namespace __llvm_libc

#endif // defined(LLVM_LIBC_ARCH_X86)

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BACKEND_X86_H
