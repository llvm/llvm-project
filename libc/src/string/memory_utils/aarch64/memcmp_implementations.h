//===-- Memcmp implementation for aarch64 -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCMP_IMPLEMENTATIONS_H
#define LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCMP_IMPLEMENTATIONS_H

#include "src/__support/macros/config.h"       // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/utils.h" // MemcmpReturnType

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_generic_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 384)) {
    if (auto value = generic::Memcmp<16>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  }
  return generic::Memcmp<16>::loop_and_tail(p1, p2, count);
}

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_aarch64_neon_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 128)) { // [128, ∞]
    if (auto value = generic::Memcmp<16>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
    return generic::Memcmp<32>::loop_and_tail(p1, p2, count);
  }
  if (generic::Bcmp<16>::block(p1, p2)) // [16, 16]
    return generic::Memcmp<16>::block(p1, p2);
  if (count < 32) // [17, 31]
    return generic::Memcmp<16>::tail(p1, p2, count);
  if (generic::Bcmp<16>::block(p1 + 16, p2 + 16)) // [32, 32]
    return generic::Memcmp<16>::block(p1 + 16, p2 + 16);
  if (count < 64) // [33, 63]
    return generic::Memcmp<32>::tail(p1, p2, count);
  // [64, 127]
  return generic::Memcmp<16>::loop_and_tail(p1 + 32, p2 + 32, count - 32);
}

LIBC_INLINE MemcmpReturnType inline_memcmp_aarch64(CPtr p1, CPtr p2,
                                                   size_t count) {
  if (count == 0)
    return MemcmpReturnType::ZERO();
  if (count == 1)
    return generic::Memcmp<1>::block(p1, p2);
  if (count == 2)
    return generic::Memcmp<2>::block(p1, p2);
  if (count == 3)
    return generic::Memcmp<3>::block(p1, p2);
  if (count <= 8)
    return generic::Memcmp<4>::head_tail(p1, p2, count);
  if (count <= 16)
    return generic::Memcmp<8>::head_tail(p1, p2, count);
  if constexpr (aarch64::kNeon)
    return inline_memcmp_aarch64_neon_gt16(p1, p2, count);
  else
    return inline_memcmp_generic_gt16(p1, p2, count);
}
} // namespace __llvm_libc

#endif // LIBC_SRC_STRING_MEMORY_UTILS_X86_64_MEMCMP_IMPLEMENTATIONS_H
