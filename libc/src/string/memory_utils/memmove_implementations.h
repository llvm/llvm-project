//===-- Memmove implementation ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_IMPLEMENTATIONS_H

#include "src/__support/common.h"
#include "src/__support/macros/optimization.h"
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include <stddef.h> // size_t, ptrdiff_t

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE void
inline_memmove_embedded_tiny(Ptr dst, CPtr src, size_t count) {
  if ((count == 0) || (dst == src))
    return;
  if (dst < src) {
    LIBC_LOOP_NOUNROLL
    for (size_t offset = 0; offset < count; ++offset)
      builtin::Memcpy<1>::block(dst + offset, src + offset);
  } else {
    LIBC_LOOP_NOUNROLL
    for (ptrdiff_t offset = count - 1; offset >= 0; --offset)
      builtin::Memcpy<1>::block(dst + offset, src + offset);
  }
}

LIBC_INLINE void inline_memmove(Ptr dst, CPtr src, size_t count) {
#if defined(LIBC_TARGET_ARCH_IS_X86) || defined(LIBC_TARGET_ARCH_IS_AARCH64)
#if defined(LIBC_TARGET_ARCH_IS_X86)
#if defined(__AVX512F__)
  using uint128_t = uint8x16_t;
  using uint256_t = uint8x32_t;
  using uint512_t = uint8x64_t;
#elif defined(__AVX__)
  using uint128_t = uint8x16_t;
  using uint256_t = uint8x32_t;
  using uint512_t = cpp::array<uint8x32_t, 2>;
#elif defined(__SSE2__)
  using uint128_t = uint8x16_t;
  using uint256_t = cpp::array<uint8x16_t, 2>;
  using uint512_t = cpp::array<uint8x16_t, 4>;
#else
  using uint128_t = cpp::array<uint64_t, 2>;
  using uint256_t = cpp::array<uint64_t, 4>;
  using uint512_t = cpp::array<uint64_t, 8>;
#endif
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  static_assert(aarch64::kNeon, "aarch64 supports vector types");
  using uint128_t = uint8x16_t;
  using uint256_t = uint8x32_t;
  using uint512_t = uint8x64_t;
#endif
  if (count == 0)
    return;
  if (count == 1)
    return generic::Memmove<uint8_t>::block(dst, src);
  if (count <= 4)
    return generic::Memmove<uint16_t>::head_tail(dst, src, count);
  if (count <= 8)
    return generic::Memmove<uint32_t>::head_tail(dst, src, count);
  if (count <= 16)
    return generic::Memmove<uint64_t>::head_tail(dst, src, count);
  if (count <= 32)
    return generic::Memmove<uint128_t>::head_tail(dst, src, count);
  if (count <= 64)
    return generic::Memmove<uint256_t>::head_tail(dst, src, count);
  if (count <= 128)
    return generic::Memmove<uint512_t>::head_tail(dst, src, count);
  if (dst < src) {
    generic::Memmove<uint256_t>::align_forward<Arg::Src>(dst, src, count);
    return generic::Memmove<uint512_t>::loop_and_tail_forward(dst, src, count);
  } else {
    generic::Memmove<uint256_t>::align_backward<Arg::Src>(dst, src, count);
    return generic::Memmove<uint512_t>::loop_and_tail_backward(dst, src, count);
  }
#else
  return inline_memmove_embedded_tiny(dst, src, count);
#endif
}

LIBC_INLINE void inline_memmove(void *dst, const void *src, size_t count) {
  inline_memmove(reinterpret_cast<Ptr>(dst), reinterpret_cast<CPtr>(src),
                 count);
}

} // namespace __llvm_libc

#endif /* LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_IMPLEMENTATIONS_H */
