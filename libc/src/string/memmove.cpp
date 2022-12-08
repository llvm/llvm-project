//===-- Implementation of memmove -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memmove.h"
#include "src/string/memory_utils/memmove_implementations.h"
#include <stddef.h> // size_t

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void *, memmove,
                   (void *dst, const void *src, size_t count)) {
  inline_memmove(dst, src, count);
  return dst;
}

} // namespace __llvm_libc
