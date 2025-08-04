//===-- Strlen implementation -----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_STRLEN_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_STRLEN_H

#include "src/__support/macros/attributes.h"               // LIBC_INLINE
#include "src/__support/macros/properties/architectures.h" // LIBC_TARGET_ARCH_IS_

#include <stddef.h> // size_t

#if defined(LIBC_COPT_STRING_UNSAFE_WIDE_READ)
#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "src/string/memory_utils/x86_64/inline_strlen.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_STRLEN_WIDE_READ string_length_x86
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "src/string/memory_utils/aarch64/inline_memcpy.h"
#define LIBC_SRC_STRING_MEMORY_UTILS_STRLEN_WIDE_READ string_length_aarch64
#else
#define LIBC_SRC_STRING_MEMORY_UTILS_STRLEN_WIDE_READ string_length_wide_read
#endif

namespace LIBC_NAMESPACE_DECL {

[[gnu::flatten]] LIBC_INLINE void
inline_memcpy(void *__restrict dst, const void *__restrict src, size_t count) {
  LIBC_SRC_STRING_MEMORY_UTILS_MEMCPY(reinterpret_cast<Ptr>(dst),
                                      reinterpret_cast<CPtr>(src), count);
}

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_INLINE_STRLEN_H
