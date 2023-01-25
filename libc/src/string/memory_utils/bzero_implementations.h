//===-- Implementation of bzero -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BZERO_IMPLEMENTATIONS_H
#define LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BZERO_IMPLEMENTATIONS_H

#include "src/__support/common.h"
#include "src/string/memory_utils/memset_implementations.h"

#include <stddef.h> // size_t

namespace __llvm_libc {

LIBC_INLINE static void inline_bzero(Ptr dst, size_t count) {
  inline_memset(dst, 0, count);
}

LIBC_INLINE static void inline_bzero(void *dst, size_t count) {
  inline_bzero(reinterpret_cast<Ptr>(dst), count);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STRING_MEMORY_UTILS_BZERO_IMPLEMENTATIONS_H
