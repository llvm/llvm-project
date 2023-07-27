//===-- Memmove implementation for x86_64 -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMMOVE_H
#define LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMMOVE_H

#include "src/__support/macros/config.h" // LIBC_INLINE
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_x86.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

LIBC_INLINE void inline_memmove_x86(Ptr dst, CPtr src, size_t count) {
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
}

} // namespace __llvm_libc

#endif // LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMMOVE_H
