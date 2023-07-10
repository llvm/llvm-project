//===-- x86 implementation of memory function building blocks -------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides x86 specific building blocks to compose memory functions.
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_X86_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_X86_H

#include "src/__support/macros/properties/architectures.h"

#if defined(LIBC_TARGET_ARCH_IS_X86_64)

#include "src/__support/common.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"

#if defined(__AVX512BW__) || defined(__AVX512F__) || defined(__AVX2__) ||      \
    defined(__SSE2__)
#include <immintrin.h>
#endif

// Define fake functions to prevent the compiler from failing on undefined
// functions in case the CPU extension is not present.
#if !defined(__AVX512BW__) && (defined(_MSC_VER) || defined(__SCE__))
#define _mm512_cmpneq_epi8_mask(A, B) 0
#endif
#if !defined(__AVX2__) && (defined(_MSC_VER) || defined(__SCE__))
#define _mm256_movemask_epi8(A) 0
#endif
#if !defined(__SSE2__) && (defined(_MSC_VER) || defined(__SCE__))
#define _mm_movemask_epi8(A) 0
#endif

namespace __llvm_libc::x86 {

// A set of constants to check compile time features.
LIBC_INLINE_VAR constexpr bool kSse2 = LLVM_LIBC_IS_DEFINED(__SSE2__);
LIBC_INLINE_VAR constexpr bool kSse41 = LLVM_LIBC_IS_DEFINED(__SSE4_1__);
LIBC_INLINE_VAR constexpr bool kAvx = LLVM_LIBC_IS_DEFINED(__AVX__);
LIBC_INLINE_VAR constexpr bool kAvx2 = LLVM_LIBC_IS_DEFINED(__AVX2__);
LIBC_INLINE_VAR constexpr bool kAvx512F = LLVM_LIBC_IS_DEFINED(__AVX512F__);
LIBC_INLINE_VAR constexpr bool kAvx512BW = LLVM_LIBC_IS_DEFINED(__AVX512BW__);

///////////////////////////////////////////////////////////////////////////////
// Memcpy repmovsb implementation
struct Memcpy {
  LIBC_INLINE static void repmovsb(void *dst, const void *src, size_t count) {
    asm volatile("rep movsb" : "+D"(dst), "+S"(src), "+c"(count) : : "memory");
  }
};

} // namespace __llvm_libc::x86

namespace __llvm_libc::generic {

///////////////////////////////////////////////////////////////////////////////
// Specializations for uint16_t
template <> struct cmp_is_expensive<uint16_t> : public cpp::false_type {};
template <> LIBC_INLINE bool eq<uint16_t>(CPtr p1, CPtr p2, size_t offset) {
  return load<uint16_t>(p1, offset) == load<uint16_t>(p2, offset);
}
template <>
LIBC_INLINE uint32_t neq<uint16_t>(CPtr p1, CPtr p2, size_t offset) {
  return load<uint16_t>(p1, offset) ^ load<uint16_t>(p2, offset);
}
template <>
LIBC_INLINE MemcmpReturnType cmp<uint16_t>(CPtr p1, CPtr p2, size_t offset) {
  return static_cast<int32_t>(load_be<uint16_t>(p1, offset)) -
         static_cast<int32_t>(load_be<uint16_t>(p2, offset));
}
template <>
LIBC_INLINE MemcmpReturnType cmp_neq<uint16_t>(CPtr p1, CPtr p2, size_t offset);

///////////////////////////////////////////////////////////////////////////////
// Specializations for uint32_t
template <> struct cmp_is_expensive<uint32_t> : public cpp::false_type {};
template <> LIBC_INLINE bool eq<uint32_t>(CPtr p1, CPtr p2, size_t offset) {
  return load<uint32_t>(p1, offset) == load<uint32_t>(p2, offset);
}
template <>
LIBC_INLINE uint32_t neq<uint32_t>(CPtr p1, CPtr p2, size_t offset) {
  return load<uint32_t>(p1, offset) ^ load<uint32_t>(p2, offset);
}
template <>
LIBC_INLINE MemcmpReturnType cmp<uint32_t>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load_be<uint32_t>(p1, offset);
  const auto b = load_be<uint32_t>(p2, offset);
  return cmp_uint32_t(a, b);
}
template <>
LIBC_INLINE MemcmpReturnType cmp_neq<uint32_t>(CPtr p1, CPtr p2, size_t offset);

///////////////////////////////////////////////////////////////////////////////
// Specializations for uint64_t
template <> struct cmp_is_expensive<uint64_t> : public cpp::true_type {};
template <> LIBC_INLINE bool eq<uint64_t>(CPtr p1, CPtr p2, size_t offset) {
  return load<uint64_t>(p1, offset) == load<uint64_t>(p2, offset);
}
template <>
LIBC_INLINE uint32_t neq<uint64_t>(CPtr p1, CPtr p2, size_t offset) {
  return !eq<uint64_t>(p1, p2, offset);
}
template <>
LIBC_INLINE MemcmpReturnType cmp<uint64_t>(CPtr p1, CPtr p2, size_t offset);
template <>
LIBC_INLINE MemcmpReturnType cmp_neq<uint64_t>(CPtr p1, CPtr p2,
                                               size_t offset) {
  const auto a = load_be<uint64_t>(p1, offset);
  const auto b = load_be<uint64_t>(p2, offset);
  return cmp_neq_uint64_t(a, b);
}

///////////////////////////////////////////////////////////////////////////////
// Specializations for __m128i
#if defined(__SSE4_1__)
template <> struct is_vector<__m128i> : cpp::true_type {};
template <> struct cmp_is_expensive<__m128i> : cpp::true_type {};
LIBC_INLINE __m128i bytewise_max(__m128i a, __m128i b) {
  return _mm_max_epu8(a, b);
}
LIBC_INLINE __m128i bytewise_reverse(__m128i value) {
  return _mm_shuffle_epi8(value, _mm_set_epi8(0, 1, 2, 3, 4, 5, 6, 7, //
                                              8, 9, 10, 11, 12, 13, 14, 15));
}
LIBC_INLINE uint16_t big_endian_cmp_mask(__m128i max, __m128i value) {
  return static_cast<uint16_t>(_mm_movemask_epi8(bytewise_reverse(_mm_cmpeq_epi8(max, value))));
}
template <> LIBC_INLINE bool eq<__m128i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m128i>(p1, offset);
  const auto b = load<__m128i>(p2, offset);
  const auto xored = _mm_xor_si128(a, b);
  return _mm_testz_si128(xored, xored) == 1; // 1 iff xored == 0
}
template <> LIBC_INLINE uint32_t neq<__m128i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m128i>(p1, offset);
  const auto b = load<__m128i>(p2, offset);
  const auto xored = _mm_xor_si128(a, b);
  return _mm_testz_si128(xored, xored) == 0; // 0 iff xored != 0
}
template <>
LIBC_INLINE MemcmpReturnType cmp_neq<__m128i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m128i>(p1, offset);
  const auto b = load<__m128i>(p2, offset);
  const auto vmax = bytewise_max(a, b);
  const auto le = big_endian_cmp_mask(vmax, b);
  const auto ge = big_endian_cmp_mask(vmax, a);
  static_assert(cpp::is_same_v<cpp::remove_cv_t<decltype(le)>, uint16_t>);
  return static_cast<int32_t>(ge) - static_cast<int32_t>(le);
}
#endif // __SSE4_1__

///////////////////////////////////////////////////////////////////////////////
// Specializations for __m256i
#if defined(__AVX__)
template <> struct is_vector<__m256i> : cpp::true_type {};
template <> struct cmp_is_expensive<__m256i> : cpp::true_type {};
template <> LIBC_INLINE bool eq<__m256i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m256i>(p1, offset);
  const auto b = load<__m256i>(p2, offset);
  const auto xored = _mm256_castps_si256(
      _mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
  return _mm256_testz_si256(xored, xored) == 1; // 1 iff xored == 0
}
template <> LIBC_INLINE uint32_t neq<__m256i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m256i>(p1, offset);
  const auto b = load<__m256i>(p2, offset);
  const auto xored = _mm256_castps_si256(
      _mm256_xor_ps(_mm256_castsi256_ps(a), _mm256_castsi256_ps(b)));
  return _mm256_testz_si256(xored, xored) == 0; // 0 iff xored != 0
}
#endif // __AVX__

#if defined(__AVX2__)
LIBC_INLINE __m256i bytewise_max(__m256i a, __m256i b) {
  return _mm256_max_epu8(a, b);
}
LIBC_INLINE __m256i bytewise_reverse(__m256i value) {
  return _mm256_shuffle_epi8(value,
                             _mm256_set_epi8(0, 1, 2, 3, 4, 5, 6, 7,         //
                                             8, 9, 10, 11, 12, 13, 14, 15,   //
                                             16, 17, 18, 19, 20, 21, 22, 23, //
                                             24, 25, 26, 27, 28, 29, 30, 31));
}
LIBC_INLINE uint32_t big_endian_cmp_mask(__m256i max, __m256i value) {
  return _mm256_movemask_epi8(bytewise_reverse(_mm256_cmpeq_epi8(max, value)));
}
template <>
LIBC_INLINE MemcmpReturnType cmp_neq<__m256i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m256i>(p1, offset);
  const auto b = load<__m256i>(p2, offset);
  const auto vmax = bytewise_max(a, b);
  const auto le = big_endian_cmp_mask(vmax, b);
  const auto ge = big_endian_cmp_mask(vmax, a);
  static_assert(cpp::is_same_v<cpp::remove_cv_t<decltype(le)>, uint32_t>);
  return cmp_uint32_t(ge, le);
}
#endif // __AVX2__

///////////////////////////////////////////////////////////////////////////////
// Specializations for __m512i
#if defined(__AVX512BW__)
template <> struct is_vector<__m512i> : cpp::true_type {};
template <> struct cmp_is_expensive<__m512i> : cpp::true_type {};
LIBC_INLINE __m512i bytewise_max(__m512i a, __m512i b) {
  return _mm512_max_epu8(a, b);
}
LIBC_INLINE __m512i bytewise_reverse(__m512i value) {
  return _mm512_shuffle_epi8(value,
                             _mm512_set_epi8(0, 1, 2, 3, 4, 5, 6, 7,         //
                                             8, 9, 10, 11, 12, 13, 14, 15,   //
                                             16, 17, 18, 19, 20, 21, 22, 23, //
                                             24, 25, 26, 27, 28, 29, 30, 31, //
                                             32, 33, 34, 35, 36, 37, 38, 39, //
                                             40, 41, 42, 43, 44, 45, 46, 47, //
                                             48, 49, 50, 51, 52, 53, 54, 55, //
                                             56, 57, 58, 59, 60, 61, 62, 63));
}
LIBC_INLINE uint64_t big_endian_cmp_mask(__m512i max, __m512i value) {
  return _mm512_cmpeq_epi8_mask(bytewise_reverse(max), bytewise_reverse(value));
}
template <> LIBC_INLINE bool eq<__m512i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m512i>(p1, offset);
  const auto b = load<__m512i>(p2, offset);
  return _mm512_cmpneq_epi8_mask(a, b) == 0;
}
template <> LIBC_INLINE uint32_t neq<__m512i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m512i>(p1, offset);
  const auto b = load<__m512i>(p2, offset);
  const uint64_t xored = _mm512_cmpneq_epi8_mask(a, b);
  return (xored >> 32) | (xored & 0xFFFFFFFF);
}
template <>
LIBC_INLINE MemcmpReturnType cmp_neq<__m512i>(CPtr p1, CPtr p2, size_t offset) {
  const auto a = load<__m512i>(p1, offset);
  const auto b = load<__m512i>(p2, offset);
  const auto vmax = bytewise_max(a, b);
  const auto le = big_endian_cmp_mask(vmax, b);
  const auto ge = big_endian_cmp_mask(vmax, a);
  static_assert(cpp::is_same_v<cpp::remove_cv_t<decltype(le)>, uint64_t>);
  return cmp_neq_uint64_t(ge, le);
}
#endif // __AVX512BW__

} // namespace __llvm_libc::generic

#endif // LIBC_TARGET_ARCH_IS_X86_64

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_OP_X86_H
