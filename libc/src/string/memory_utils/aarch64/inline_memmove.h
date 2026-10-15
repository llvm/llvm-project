//===-- Memmove implementation for aarch64 ----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_MEMMOVE_H
#define LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_MEMMOVE_H

#include "src/__support/macros/attributes.h" // LIBC_INLINE
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {

LIBC_INLINE bool inline_memmove_small_size_aarch64(Ptr dst, CPtr src,
                                                   size_t count) {
#if defined(__ARM_NEON)
  using uint128_t = generic_v128;
  using uint256_t = generic_v256;
  using uint512_t = generic_v512;
#else
  using uint128_t = cpp::array<uint64_t, 2>;
  using uint256_t = cpp::array<uint64_t, 4>;
#endif
  if (count == 0)
    return true;
  if (count == 1) {
    generic::Memmove<uint8_t>::block(dst, src);
    return true;
  }
  if (count <= 4) {
    generic::Memmove<uint16_t>::head_tail(dst, src, count);
    return true;
  }
  if (count <= 8) {
    generic::Memmove<uint32_t>::head_tail(dst, src, count);
    return true;
  }
  if (count <= 16) {
    generic::Memmove<uint64_t>::head_tail(dst, src, count);
    return true;
  }
  if (count <= 32) {
    generic::Memmove<uint128_t>::head_tail(dst, src, count);
    return true;
  }
  if (count <= 64) {
    generic::Memmove<uint256_t>::head_tail(dst, src, count);
    return true;
  }
#if defined(__ARM_NEON)
  if (count <= 128) {
    generic::Memmove<uint512_t>::head_tail(dst, src, count);
    return true;
  }
#endif
  return false;
}

LIBC_INLINE void inline_memmove_follow_up_aarch64(Ptr dst, CPtr src,
                                                  size_t count) {
#if defined(__ARM_NEON)
  using AlignT = generic_v256;
  using LoopT = generic_v512;
#else
  using AlignT = cpp::array<uint64_t, 2>;
  using LoopT = cpp::array<uint64_t, 4>;
#endif
  if (dst < src) {
    generic::Memmove<AlignT>::align_forward<Arg::Src>(dst, src, count);
    return generic::Memmove<LoopT>::loop_and_tail_forward(dst, src, count);
  } else {
    generic::Memmove<AlignT>::align_backward<Arg::Src>(dst, src, count);
    return generic::Memmove<LoopT>::loop_and_tail_backward(dst, src, count);
  }
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_MEMMOVE_H
