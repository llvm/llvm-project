//===-- Strlen implementation for x86_64 ----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_STRLEN_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_STRLEN_H

#include "src/__support/CPP/bit.h"          // countr_zero

#include <immintrin.h>
#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {

namespace sse2 {
[[maybe_unused]] LIBC_INLINE size_t string_length(const char *src) {
  using Vector __attribute__((may_alias)) = __m128i;

  Vector z = _mm_setzero_si128();
  uintptr_t misalign_bytes = reinterpret_cast<uintptr_t>(src) % sizeof(Vector);
  const Vector *block_ptr =
      reinterpret_cast<const Vector *>(src - misalign_bytes);
  Vector v = _mm_load_si128(block_ptr);
  Vector vcmp = _mm_cmpeq_epi8(z, v);
  // shift away results in irrelevant bytes.
  uint32_t cmp = _mm_movemask_epi8(vcmp) >> misalign_bytes;
  if (cmp)
    return cpp::countr_zero(cmp);

  while (true) {
    block_ptr++;
    v = _mm_load_si128(block_ptr);
    vcmp = _mm_cmpeq_epi8(z, v);
    cmp = _mm_movemask_epi8(vcmp);
    if (cmp)
      return static_cast<size_t>(reinterpret_cast<uintptr_t>(block_ptr) -
                                 reinterpret_cast<uintptr_t>(src) +
                                 cpp::countr_zero(cmp));
  }
}
}  // namespace sse2

#if defined(__AVX2__)
namespace avx2 {
[[maybe_unused]] LIBC_INLINE size_t string_length(const char *src) {
  using Vector __attribute__((may_alias)) = __mm256i;

  Vector z = _mm256_setzero_si256();
  uintptr_t misalign_bytes = reinterpret_cast<uintptr_t>(src) % sizeof(Vector);
  const Vector *block_ptr =
      reinterpret_cast<const Vector *>(src - misalign_bytes);
  Vector v = _mm256_load_si256(block_ptr);
  Vector vcmp = _mm256_cmpeq_epi8(z, v);
  // shift away results in irrelevant bytes.
  int cmp = _mm256_movemask_epi8(vcmp) >> misalign_bytes;
  if (cmp)
    return cpp::countr_zero(cmp);

  while (true) {
    block_ptr++;
    v = _mm256_load_si256(block_ptr);
    vcmp = _mm256_cmpeq_epi8(z, v);
    cmp = _mm256_movemask_epi8(vcmp);
    if (cmp)
      return static_cast<size_t>(reinterpret_cast<uintptr_t>(block_ptr) -
                                 reinterpret_cast<uintptr_t>(src) +
                                 cpp::countr_zero(cmp));
  }
}
}
#endif

#if defined(__AVX512F__)
namespace avx512 {
[[maybe_unused]] LIBC_INLINE size_t string_length(const char *src) {
  using Vector __attribute__((may_alias)) = __mm512i;

  Vector z = _mm512_setzero_si512();
  uintptr_t misalign_bytes = reinterpret_cast<uintptr_t>(src) % sizeof(Vector);
  const Vector *block_ptr =
      reinterpret_cast<const Vector *>(src - misalign_bytes);
  Vector v = _mm512_load_si512(block_ptr);
  __mmask64 cmp = _mm512_cmp_epu8_mask(z, v, _MM_CMPINT_EQ) >> misalign_bytes;
  if (cmp)
    return cpp::countr_zero(cmp);

  while (true) {
    block_ptr++;
    Vector v = _mm512_load_si512(block_ptr);
    __mmask64 cmp = _mm512_cmp_epu8_mask(z, v, _MM_CMPINT_EQ);
    if (cmp)
      return static_cast<size_t>(reinterpret_cast<uintptr_t>(block_ptr) -
                                 reinterpret_cast<uintptr_t>(src) +
                                 cpp::countr_zero(cmp));
  }
}
} // namespace avx512
#endif

#if defined(__AVX512F__)
namespace string_length_impl = avx512;
#elif defined(__AVX2__)
namespace string_length_impl = avx2;
#else
namespace string_length_impl = sse2;
#endif

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_STRLEN_H
