//===-- Implementation of memcmp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memcmp.h"
#include "src/string/memory_utils/inline_memcmp.h"

#include <stddef.h> // size_t

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, memcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  return inline_memcmp(lhs, rhs, count);
}

} // namespace LIBC_NAMESPACE
