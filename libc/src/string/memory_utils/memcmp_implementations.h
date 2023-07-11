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
#include "src/string/memory_utils/generic/aligned_access.h"
#include "src/string/memory_utils/generic/byte_per_byte.h"
#include "src/string/memory_utils/op_generic.h"
#include "src/string/memory_utils/op_riscv.h"
#include "src/string/memory_utils/utils.h" // CPtr MemcmpReturnType

#include <stddef.h> // size_t

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/memcmp_implementations.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/memcmp_implementations.h"
#endif

namespace __llvm_libc {

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
