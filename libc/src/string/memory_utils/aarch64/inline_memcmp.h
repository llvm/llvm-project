//===-- Memcmp implementation for aarch64 -----------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMCMP_H
#define LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMCMP_H

#include "src/__support/macros/config.h"       // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY
#include "src/string/memory_utils/op_aarch64.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/utils.h" // MemcmpReturnType

namespace LIBC_NAMESPACE {

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_generic_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 384)) {
    if (auto value = generic::Memcmp<uint8x16_t>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
  }
  return generic::Memcmp<uint8x16_t>::loop_and_tail(p1, p2, count);
}

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_aarch64_neon_gt16(CPtr p1, CPtr p2, size_t count) {
  if (LIBC_UNLIKELY(count >= 128)) { // [128, ∞]
    if (auto value = generic::Memcmp<uint8x16_t>::block(p1, p2))
      return value;
    align_to_next_boundary<16, Arg::P1>(p1, p2, count);
    return generic::Memcmp<uint8x16x2_t>::loop_and_tail(p1, p2, count);
  }
  if (generic::Bcmp<uint8x16_t>::block(p1, p2)) // [16, 16]
    return generic::Memcmp<uint8x16_t>::block(p1, p2);
  if (count < 32) // [17, 31]
    return generic::Memcmp<uint8x16_t>::tail(p1, p2, count);
  if (generic::Bcmp<uint8x16_t>::block(p1 + 16, p2 + 16)) // [32, 32]
    return generic::Memcmp<uint8x16_t>::block(p1 + 16, p2 + 16);
  if (count < 64) // [33, 63]
    return generic::Memcmp<uint8x16x2_t>::tail(p1, p2, count);
  // [64, 127]
  return generic::Memcmp<uint8x16_t>::loop_and_tail(p1 + 32, p2 + 32,
                                                    count - 32);
}

LIBC_INLINE MemcmpReturnType inline_memcmp_aarch64(CPtr p1, CPtr p2,
                                                   size_t count) {
  if (count == 0)
    return MemcmpReturnType::ZERO();
  if (count == 1)
    return generic::Memcmp<uint8_t>::block(p1, p2);
  if (count == 2)
    return generic::Memcmp<uint16_t>::block(p1, p2);
  if (count == 3)
    return generic::MemcmpSequence<uint16_t, uint8_t>::block(p1, p2);
  if (count <= 8)
    return generic::Memcmp<uint32_t>::head_tail(p1, p2, count);
  if (count <= 16)
    return generic::Memcmp<uint64_t>::head_tail(p1, p2, count);
  if constexpr (aarch64::kNeon)
    return inline_memcmp_aarch64_neon_gt16(p1, p2, count);
  else
    return inline_memcmp_generic_gt16(p1, p2, count);
}
} // namespace LIBC_NAMESPACE

#endif // LIBC_SRC_STRING_MEMORY_UTILS_X86_64_INLINE_MEMCMP_H
