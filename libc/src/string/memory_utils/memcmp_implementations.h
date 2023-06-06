//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H

#include "src/__support/common.h"
#include "src/__support/macros/optimization.h" // LIBC_UNLIKELY LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/utils.h" // CPtr MemcmpReturnType

#include <stddef.h> // size_t

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/memcmp_implementations.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/memcmp_implementations.h"
#endif

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_byte_per_byte(CPtr p1, CPtr p2, size_t count, size_t offset = 0) {
  return generic::Memcmp<uint8_t>::loop_and_tail_offset(p1, p2, count, offset);
}

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_aligned_access_64bit(CPtr p1, CPtr p2, size_t count) {
  constexpr size_t kAlign = sizeof(uint64_t);
  if (count <= 2 * kAlign)
    return inline_memcmp_byte_per_byte(p1, p2, count);
  size_t bytes_to_p1_align = distance_to_align_up<kAlign>(p1);
  if (auto value = inline_memcmp_byte_per_byte(p1, p2, bytes_to_p1_align))
    return value;
  size_t offset = bytes_to_p1_align;
  size_t p2_alignment = distance_to_align_down<kAlign>(p2 + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint64_t b;
    if (p2_alignment == 0)
      b = load64_aligned<uint64_t>(p2, offset);
    else if (p2_alignment == 4)
      b = load64_aligned<uint32_t, uint32_t>(p2, offset);
    else if (p2_alignment == 2)
      b = load64_aligned<uint16_t, uint16_t, uint16_t, uint16_t>(p2, offset);
    else
      b = load64_aligned<uint8_t, uint16_t, uint16_t, uint16_t, uint8_t>(
          p2, offset);
    uint64_t a = load64_aligned<uint64_t>(p1, offset);
    if (a != b)
      return cmp_neq_uint64_t(Endian::to_big_endian(a),
                              Endian::to_big_endian(b));
  }
  return inline_memcmp_byte_per_byte(p1, p2, count, offset);
}

[[maybe_unused]] LIBC_INLINE MemcmpReturnType
inline_memcmp_aligned_access_32bit(CPtr p1, CPtr p2, size_t count) {
  constexpr size_t kAlign = sizeof(uint32_t);
  if (count <= 2 * kAlign)
    return inline_memcmp_byte_per_byte(p1, p2, count);
  size_t bytes_to_p1_align = distance_to_align_up<kAlign>(p1);
  if (auto value = inline_memcmp_byte_per_byte(p1, p2, bytes_to_p1_align))
    return value;
  size_t offset = bytes_to_p1_align;
  size_t p2_alignment = distance_to_align_down<kAlign>(p2 + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint32_t b;
    if (p2_alignment == 0)
      b = load32_aligned<uint32_t>(p2, offset);
    else if (p2_alignment == 2)
      b = load32_aligned<uint16_t, uint16_t>(p2, offset);
    else
      b = load32_aligned<uint8_t, uint16_t, uint8_t>(p2, offset);
    uint32_t a = load32_aligned<uint32_t>(p1, offset);
    if (a != b)
      return cmp_uint32_t(Endian::to_big_endian(a), Endian::to_big_endian(b));
  }
  return inline_memcmp_byte_per_byte(p1, p2, count, offset);
}

LIBC_INLINE MemcmpReturnType inline_memcmp(CPtr p1, CPtr p2, size_t count) {
#if defined(LIBC_TARGET_ARCH_IS_X86)
  return inline_memcmp_x86(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  return inline_memcmp_aarch64(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV64)
  return inline_memcmp_aligned_access_64bit(p1, p2, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV32)
  return inline_memcmp_aligned_access_32bit(p1, p2, count);
#else
  return inline_memcmp_byte_per_byte(p1, p2, count);
#endif
}

LIBC_INLINE int inline_memcmp(const void *p1, const void *p2, size_t count) {
  return static_cast<int>(inline_memcmp(reinterpret_cast<CPtr>(p1),
                                        reinterpret_cast<CPtr>(p2), count));
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCMP_IMPLEMENTATIONS_H
