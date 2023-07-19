//===-- Implementation of memset ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/memset.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/inline_memset.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void *, memset, (void *dst, int value, size_t count)) {
  inline_memset(dst, static_cast<uint8_t>(value), count);
  return dst;
}

} // namespace __llvm_libc
