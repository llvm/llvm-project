//===-- Memcpy implementation -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H

#include "src/__support/macros/config.h"       // LIBC_INLINE
#include "src/__support/macros/optimization.h" // LIBC_LOOP_NOUNROLL
#include "src/__support/macros/properties/architectures.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/memcpy_implementations.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/memcpy_implementations.h"
#endif

namespace __llvm_libc {

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_byte_per_byte(Ptr dst, CPtr src, size_t offset, size_t count) {
  LIBC_LOOP_NOUNROLL
  for (; offset < count; ++offset)
    dst[offset] = src[offset];
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_aligned_access_32bit(Ptr __restrict dst, CPtr __restrict src,
                                   size_t count) {
  constexpr size_t kAlign = sizeof(uint32_t);
  if (count <= 2 * kAlign)
    return inline_memcpy_byte_per_byte(dst, src, 0, count);
  size_t bytes_to_dst_align = distance_to_align_up<kAlign>(dst);
  inline_memcpy_byte_per_byte(dst, src, 0, bytes_to_dst_align);
  size_t offset = bytes_to_dst_align;
  size_t src_alignment = distance_to_align_down<kAlign>(src + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint32_t value;
    if (src_alignment == 0)
      value = load32_aligned<uint32_t>(src, offset);
    else if (src_alignment == 2)
      value = load32_aligned<uint16_t, uint16_t>(src, offset);
    else
      value = load32_aligned<uint8_t, uint16_t, uint8_t>(src, offset);
    store32_aligned<uint32_t>(value, dst, offset);
  }
  // remainder
  inline_memcpy_byte_per_byte(dst, src, offset, count);
}

[[maybe_unused]] LIBC_INLINE void
inline_memcpy_aligned_access_64bit(Ptr __restrict dst, CPtr __restrict src,
                                   size_t count) {
  constexpr size_t kAlign = sizeof(uint64_t);
  if (count <= 2 * kAlign)
    return inline_memcpy_byte_per_byte(dst, src, 0, count);
  size_t bytes_to_dst_align = distance_to_align_up<kAlign>(dst);
  inline_memcpy_byte_per_byte(dst, src, 0, bytes_to_dst_align);
  size_t offset = bytes_to_dst_align;
  size_t src_alignment = distance_to_align_down<kAlign>(src + offset);
  for (; offset < count - kAlign; offset += kAlign) {
    uint64_t value;
    if (src_alignment == 0)
      value = load64_aligned<uint64_t>(src, offset);
    else if (src_alignment == 4)
      value = load64_aligned<uint32_t, uint32_t>(src, offset);
    else if (src_alignment == 2)
      value =
          load64_aligned<uint16_t, uint16_t, uint16_t, uint16_t>(src, offset);
    else
      value = load64_aligned<uint8_t, uint16_t, uint16_t, uint16_t, uint8_t>(
          src, offset);
    store64_aligned<uint64_t>(value, dst, offset);
  }
  // remainder
  inline_memcpy_byte_per_byte(dst, src, offset, count);
}

LIBC_INLINE void inline_memcpy(Ptr __restrict dst, CPtr __restrict src,
                               size_t count) {
  using namespace __llvm_libc::builtin;
#if defined(LIBC_COPT_MEMCPY_USE_EMBEDDED_TINY)
  return inline_memcpy_byte_per_byte(dst, src, 0, count);
#elif defined(LIBC_TARGET_ARCH_IS_X86)
  return inline_memcpy_x86_maybe_interpose_repmovsb(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  return inline_memcpy_aarch64(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV64)
  return inline_memcpy_aligned_access_64bit(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV32)
  return inline_memcpy_aligned_access_32bit(dst, src, count);
#else
  return inline_memcpy_byte_per_byte(dst, src, 0, count);
#endif
}

LIBC_INLINE void inline_memcpy(void *__restrict dst, const void *__restrict src,
                               size_t count) {
  inline_memcpy(reinterpret_cast<Ptr>(dst), reinterpret_cast<CPtr>(src), count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
