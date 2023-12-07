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
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE inline_memmove_x86
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/inline_memmove.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE inline_memmove_aarch64
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "src/string/memory_utils/riscv/inline_memmove.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE inline_memmove_riscv
#elif defined(LIBC_TARGET_ARCH_IS_ARM) || defined(LIBC_TARGET_ARCH_IS_GPU)
#include "src/string/memory_utils/generic/byte_per_byte.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE inline_memmove_byte_per_byte
#else
#error "Unsupported architecture"
#endif

namespace __llvm_libc {

LIBC_INLINE void inline_memmove(void *dst, const void *src, size_t count) {
  LIBC_SRC_STRING_MEMORY_UTILS_MEMMOVE(reinterpret_cast<Ptr>(dst),
                                       reinterpret_cast<CPtr>(src), count);
}

} // namespace __llvm_libc

#endif /* LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_MEMMOVE_H */
