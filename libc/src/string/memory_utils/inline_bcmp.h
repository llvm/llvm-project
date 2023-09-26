//===-- Dispatch logic for bcmp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_BCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_BCMP_H

#include "src/__support/common.h"
#include "src/__support/macros/properties/architectures.h" // LIBC_TARGET_ARCH_IS_

#include <stddef.h> // size_t

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/inline_bcmp.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_x86
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/inline_bcmp.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_aarch64
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "src/string/memory_utils/riscv/inline_bcmp.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_riscv
#elif defined(LIBC_TARGET_ARCH_IS_ARM) || defined(LIBC_TARGET_ARCH_IS_GPU)
#include "src/string/memory_utils/generic/byte_per_byte.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_byte_per_byte
#else
#error "Unsupported architecture"
#endif

namespace LIBC_NAMESPACE {

LIBC_INLINE int inline_bcmp(const void *p1, const void *p2, size_t count) {
  return static_cast<int>(LIBC_SRC_STRING_MEMORY_UTILS_BCMP(
      reinterpret_cast<CPtr>(p1), reinterpret_cast<CPtr>(p2), count));
}

} // namespace LIBC_NAMESPACE

#undef LIBC_SRC_STRING_MEMORY_UTILS_BCMP

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_BCMP_H
