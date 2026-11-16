//===-- Dispatch logic for bcmp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_BCMP_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_BCMP_H

#include "src/__support/macros/attributes.h" // LIBC_ALWAYS_INLINE, LIBC_FLATTEN
#include "src/__support/macros/config.h"     // LIBC_NAMESPACE_DECL
#include "src/__support/macros/properties/architectures.h" // LIBC_TARGET_ARCH_IS_
#include "src/string/memory_utils/utils.h"                 // CPtr

#include <stddef.h> // size_t

#if defined(LIBC_COPT_USE_MEM_BUILTINS)
#include "src/string/memory_utils/generic/builtin.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_builtin
#elif defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/inline_bcmp.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_x86
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#if defined(__ARM_FEATURE_UNALIGNED) || defined(_M_ARM64)
#include "src/string/memory_utils/aarch64/inline_bcmp.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_aarch64
#else
#include "src/string/memory_utils/generic/aligned_access.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_aligned_access_64bit
#endif
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
#if defined(__ARM_FEATURE_UNALIGNED)
#include "src/string/memory_utils/generic/bcmp_utils.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_unaligned_32bit
#else
#include "src/string/memory_utils/generic/aligned_access.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_aligned_access_32bit
#endif
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "src/string/memory_utils/riscv/inline_bcmp.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_riscv
#else
#include "src/string/memory_utils/generic/byte_per_byte.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_BCMP inline_bcmp_byte_per_byte
#endif

namespace LIBC_NAMESPACE_DECL {

// Always inline so `bcmp` (aligned to 64 bytes) does not emit a tail-call stub
// to an unaligned out-of-line `inline_bcmp`.
LIBC_FLATTEN LIBC_ALWAYS_INLINE int inline_bcmp(const void *p1, const void *p2,
                                                size_t count) {
  return static_cast<int>(LIBC_SRC_STRING_MEMORY_UTILS_BCMP(
      reinterpret_cast<CPtr>(p1), reinterpret_cast<CPtr>(p2), count));
}

} // namespace LIBC_NAMESPACE_DECL

#undef LIBC_SRC_STRING_MEMORY_UTILS_BCMP

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_BCMP_H
