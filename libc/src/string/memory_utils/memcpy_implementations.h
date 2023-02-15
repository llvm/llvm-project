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
inline_memcpy_embedded_tiny(Ptr __restrict dst, CPtr __restrict src,
                            size_t count) {
  LIBC_LOOP_NOUNROLL
  for (size_t offset = 0; offset < count; ++offset)
    builtin::Memcpy<1>::block(dst + offset, src + offset);
}

LIBC_INLINE void inline_memcpy(Ptr __restrict dst, CPtr __restrict src,
                               size_t count) {
  using namespace __llvm_libc::builtin;
#if defined(LIBC_COPT_MEMCPY_USE_EMBEDDED_TINY) ||                             \
    defined(LIBC_TARGET_ARCH_IS_ARM) || defined(LIBC_TARGET_ARCH_IS_GPU)
  return inline_memcpy_embedded_tiny(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_X86)
  return inline_memcpy_x86_maybe_interpose_repmovsb(dst, src, count);
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
  return inline_memcpy_aarch64(dst, src, count);
#else
#error "Unsupported platform"
#endif
}

LIBC_INLINE void inline_memcpy(void *__restrict dst, const void *__restrict src,
                               size_t count) {
  inline_memcpy(reinterpret_cast<Ptr>(dst), reinterpret_cast<CPtr>(src), count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY_IMPLEMENTATIONS_H
