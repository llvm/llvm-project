//===-- Memset implementation for x86_64 ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMSET_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMSET_H

#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h" // Ptr, CPtr
#include "third_party/llvm/llvm-project/libc/src/string/memory_utils/inline_memcpy.h"

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE {
namespace x86 {
// Size of one cache line for software prefetching
LIBC_INLINE_VAR constexpr size_t kOneCachelineSize = 64;
LIBC_INLINE_VAR constexpr size_t kTwoCachelinesSize = kOneCachelineSize * 2;
LIBC_INLINE_VAR constexpr size_t kFiveCachelinesSize = kOneCachelineSize * 5;

LIBC_INLINE_VAR constexpr bool kUseSoftwarePrefetchingMemset =
    LLVM_LIBC_IS_DEFINED(LIBC_COPT_MEMSET_X86_USE_SOFTWARE_PREFETCHING);

} // namespace x86

#if defined(__AVX512F__)
using uint128_t = generic_v128;
using uint256_t = generic_v256;
using uint512_t = generic_v512;
#elif defined(__AVX__)
using uint128_t = generic_v128;
using uint256_t = generic_v256;
using uint512_t = cpp::array<generic_v256, 2>;
#elif defined(__SSE2__)
using uint128_t = generic_v128;
using uint256_t = cpp::array<generic_v128, 2>;
using uint512_t = cpp::array<generic_v128, 4>;
#else
using uint128_t = cpp::array<uint64_t, 2>;
using uint256_t = cpp::array<uint64_t, 4>;
using uint512_t = cpp::array<uint64_t, 8>;
#endif

[[maybe_unused]] LIBC_INLINE static void
inline_memset_x86_gt64_sw_prefetching(Ptr dst, uint8_t value, size_t count) {
  size_t prefetch_distance = x86::kFiveCachelinesSize;
  size_t prefetch_degree = x86::kTwoCachelinesSize;
  size_t SIZE = sizeof(uint256_t);
  // Prefetch one cache line
  prefetch_for_write(dst + x86::kOneCachelineSize);
  if (count <= 128)
    return generic::Memset<uint512_t>::head_tail(dst, value, count);
  // Prefetch the second cache line
  prefetch_for_write(dst + x86::kTwoCachelinesSize);
  // Aligned loop
  generic::Memset<uint256_t>::block(dst, value);
  align_to_next_boundary<32>(dst, count);
  if (count <= 192) {
    return generic::Memset<uint256_t>::loop_and_tail(dst, value, count);
  } else {
    generic::Memset<uint512_t>::block(dst, value);
    generic::Memset<uint256_t>::block_offset(dst, value, SIZE);
    size_t offset = 96;
    while (offset + prefetch_degree + SIZE <= count) {
      for (size_t i = 0; i < prefetch_degree / x86::kOneCachelineSize; ++i)
        prefetch_for_write(dst + offset + prefetch_distance +
                           x86::kOneCachelineSize * i);
      for (size_t i = 0; i < prefetch_degree; i += SIZE, offset += SIZE)
        generic::Memset<uint256_t>::block_offset(dst, value, offset);
    }
    generic::Memset<uint256_t>::loop_and_tail_offset(dst, value, count, offset);
  }
}

[[maybe_unused]] LIBC_INLINE static void
inline_memset_x86(Ptr dst, uint8_t value, size_t count) {
  if (count == 0)
    return;
  if (count == 1)
    return generic::Memset<uint8_t>::block(dst, value);
  if (count == 2)
    return generic::Memset<uint16_t>::block(dst, value);
  if (count == 3)
    return generic::MemsetSequence<uint16_t, uint8_t>::block(dst, value);
  if (count <= 8)
    return generic::Memset<uint32_t>::head_tail(dst, value, count);
  if (count <= 16)
    return generic::Memset<uint64_t>::head_tail(dst, value, count);
  if (count <= 32)
    return generic::Memset<uint128_t>::head_tail(dst, value, count);
  if (count <= 64)
    return generic::Memset<uint256_t>::head_tail(dst, value, count);
  if constexpr (x86::kUseSoftwarePrefetchingMemset)
    return inline_memset_x86_gt64_sw_prefetching(dst, value, count);
  if (count <= 128)
    return generic::Memset<uint512_t>::head_tail(dst, value, count);
  // Aligned loop
  generic::Memset<uint256_t>::block(dst, value);
  align_to_next_boundary<32>(dst, count);
  return generic::Memset<uint256_t>::loop_and_tail(dst, value, count);
}
} // namespace LIBC_NAMESPACE

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMSET_H
