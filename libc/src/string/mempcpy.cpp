//===-- Implementation of mempcpy ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/mempcpy.h"
#include "src/string/memory_utils/inline_memcpy.h"

#include "src/__support/common.h"
#include <stddef.h> // For size_t.

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void *, mempcpy,
                   (void *__restrict dst, const void *__restrict src,
                    size_t count)) {
  inline_memcpy(dst, src, count);
  return reinterpret_cast<char *>(dst) + count;
}

} // namespace LIBC_NAMESPACE
