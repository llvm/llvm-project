//===-- Strlen implementation for x86_64 ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_STRLEN_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_STRLEN_H

#include "src/__support/CPP/bit.h" // countr_zero

#include <immintrin.h>
#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {

namespace string_length_internal {
// Return a bit-mask with the nth bit set if the nth-byte in block_ptr is zero.
template <typename Vector, typename Mask>
LIBC_INLINE static Mask compare_and_mask(const Vector *block_ptr);

template <typename Vector, typename Mask,
          decltype(compare_and_mask<Vector, Mask>)>
size_t string_length_vector(const char *src) {
  uintptr_t misalign_bytes = reinterpret_cast<uintptr_t>(src) % sizeof(Vector);

  const Vector *block_ptr =
      reinterpret_cast<const Vector *>(src - misalign_bytes);
  auto cmp = compare_and_mask<Vector, Mask>(block_ptr) >> misalign_bytes;
  if (cmp)
    return cpp::countr_zero(cmp);

  while (true) {
    block_ptr++;
    cmp = compare_and_mask<Vector, Mask>(block_ptr);
    if (cmp)
      return static_cast<size_t>(reinterpret_cast<uintptr_t>(block_ptr) -
                                 reinterpret_cast<uintptr_t>(src) +
                                 cpp::countr_zero(cmp));
  }
}

template <>
LIBC_INLINE uint32_t
compare_and_mask<__m128i, uint32_t>(const __m128i *block_ptr) {
  __m128i v = _mm_load_si128(block_ptr);
  __m128i z = _mm_setzero_si128();
  __m128i c = _mm_cmpeq_epi8(z, v);
  return _mm_movemask_epi8(c);
}

namespace sse2 {
[[maybe_unused]] LIBC_INLINE size_t string_length(const char *src) {
  return string_length_vector<__m128i, uint32_t,
                              compare_and_mask<__m128i, uint32_t>>(src);
}
} // namespace sse2

#if defined(__AVX2__)
template <>
LIBC_INLINE uint32_t
compare_and_mask<__m256i, uint32_t>(const __m256i *block_ptr) {
  __m256i v = _mm256_load_si256(block_ptr);
  __m256i z = _mm256_setzero_si256();
  __m256i c = _mm256_cmpeq_epi8(z, v);
  return _mm256_movemask_epi8(c);
}

namespace avx2 {
[[maybe_unused]] LIBC_INLINE size_t string_length(const char *src) {
  return string_length_vector<__m256i, uint32_t,
                              compare_and_mask<__m256i, uint32_t>>(src);
}
} // namespace avx2
#endif

#if defined(__AVX512F__)
template <>
LIBC_INLINE __mmask64
compare_and_mask<__m512i, __mmask64>(const __m512i *block_ptr) {
  __m512i v = _mm512_load_si512(block_ptr);
  __m512i z = _mm512_setzero_si512();
  return _mm512_cmp_epu8_mask(z, v, _MM_CMPINT_EQ);
}
namespace avx512 {
[[maybe_unused]] LIBC_INLINE size_t string_length(const char *src) {
  return string_length_vector<__m512i, __mmask64,
                              compare_and_mask<__m512i, __mmask64>>(src);
}
} // namespace avx512
#endif
} // namespace string_length_internal

#if defined(__AVX512F__)
namespace string_length_impl = string_length_internal::avx512;
#elif defined(__AVX2__)
namespace string_length_impl = string_length_internal::avx2;
#else
namespace string_length_impl = string_length_internal::sse2;
#endif

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_STRLEN_H
