//===-- Memmove implementation ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_MEMMOVE_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_MEMMOVE_H

#include <stddef.h> // size_t, ptrdiff_t

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/inline_memmove.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_SMALL_SIZE                        \
  inline_memmove_small_size_x86
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_FOLLOW_UP                         \
  inline_memmove_follow_up_x86
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/inline_memmove.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_SMALL_SIZE                        \
  inline_memmove_no_small_size
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_FOLLOW_UP inline_memmove_aarch64
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "src/string/memory_utils/riscv/inline_memmove.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_SMALL_SIZE                        \
  inline_memmove_no_small_size
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_FOLLOW_UP inline_memmove_riscv
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
#include "src/string/memory_utils/generic/byte_per_byte.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_SMALL_SIZE                        \
  inline_memmove_no_small_size
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_FOLLOW_UP                         \
  inline_memmove_byte_per_byte
#elif defined(LIBC_TARGET_ARCH_IS_GPU)
#include "src/string/memory_utils/generic/builtin.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_SMALL_SIZE                        \
  inline_memmove_no_small_size
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_FOLLOW_UP inline_memmove_builtin
#else
#error "Unsupported architecture"
#endif

namespace LIBC_NAMESPACE {

LIBC_INLINE constexpr bool inline_memmove_no_small_size(void *, const void *,
                                                        size_t) {
  return false;
}

LIBC_INLINE bool inline_memmove_small_size(void *dst, const void *src,
                                           size_t count) {
  return LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_SMALL_SIZE(
      reinterpret_cast<Ptr>(dst), reinterpret_cast<CPtr>(src), count);
}

LIBC_INLINE void inline_memmove_follow_up(void *dst, const void *src,
                                          size_t count) {
  LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE_FOLLOW_UP(
      reinterpret_cast<Ptr>(dst), reinterpret_cast<CPtr>(src), count);
}

LIBC_INLINE void inline_memmove(void *dst, const void *src, size_t count) {
  if (inline_memmove_small_size(dst, src, count))
    return;
  inline_memmove_follow_up(dst, src, count);
}

} // namespace LIBC_NAMESPACE

#endif /* LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_MEMMOVE_H */
