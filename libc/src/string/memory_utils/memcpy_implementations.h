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
#include "src/string/memory_utils/generic/aligned_access.h"
#include "src/string/memory_utils/generic/byte_per_byte.h"
#include "src/string/memory_utils/op_builtin.h"
#include "src/string/memory_utils/utils.h"

#include <stddef.h> // size_t

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/memcpy_implementations.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/memcpy_implementations.h"
#endif

namespace __llvm_libc {

LIBC_INLINE void inline_memcpy(Ptr __restrict dst, CPtr __restrict src,
                               size_t count) {
  using namespace __llvm_libc::builtin;
#if defined(LIBC_COPT_MEMCPY_USE_EMBEDDED_TINY)
  return inline_memcpy_byte_per_byte(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_X86)
  return inline_memcpy_x86_maybe_interpose_repmovsb(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  return inline_memcpy_aarch64(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV64)
  return inline_memcpy_aligned_access_64bit(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_RISCV32)
  return inline_memcpy_aligned_access_32bit(dst, src, count);
#else
  return inline_memcpy_byte_per_byte(dst, src, count);
#endif
}

LIBC_INLINE void inline_memcpy(void *__restrict dst, const void *__restrict src,
                               size_t count) {
  inline_memcpy(reinterpret_cast<Ptr>(dst), reinterpret_cast<CPtr>(src), count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
