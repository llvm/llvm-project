//===-- Strlen implementation for aarch64 ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_STRLEN_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_STRLEN_H

#if defined(__ARM_NEON)
#include "src/__support/CPP/bit.h" // countr_zero

#include <arm_neon.h>
#include <stddef.h> // size_t

namespace LIBC_NAMESPACE_DECL {

namespace neon {
[[gnu::no_sanitize_address]] [[maybe_unused]] LIBC_INLINE static size_t
string_length(const char *src) {
  using Vector __attribute__((may_alias)) = uint8x8_t;

  uintptr_t misalign_bytes = reinterpret_cast<uintptr_t>(src) % sizeof(Vector);
  const Vector *block_ptr =
      reinterpret_cast<const Vector *>(src - misalign_bytes);
  Vector v = *block_ptr;
  Vector vcmp = vceqz_u8(v);
  uint64x1_t cmp_mask = vreinterpret_u64_u8(vcmp);
  uint64_t cmp = vget_lane_u64(cmp_mask, 0);
  cmp = cmp >> (misalign_bytes << 3);
  if (cmp)
    return cpp::countr_zero(cmp) >> 3;

  while (true) {
    ++block_ptr;
    v = *block_ptr;
    vcmp = vceqz_u8(v);
    cmp_mask = vreinterpret_u64_u8(vcmp);
    cmp = vget_lane_u64(cmp_mask, 0);
    if (cmp)
      return static_cast<size_t>(reinterpret_cast<uintptr_t>(block_ptr) -
                                 reinterpret_cast<uintptr_t>(src) +
                                 (cpp::countr_zero(cmp) >> 3));
  }
}
} // namespace neon

namespace string_length_impl = neon;

} // namespace LIBC_NAMESPACE_DECL
#endif // __ARM_NEON
#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_AARCH64_INLINE_STRLEN_H
