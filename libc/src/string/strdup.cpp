//===-- Implementation of strdup ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strdup.h"
#include "src/string/memory_utils/memcpy_implementations.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

#include <stdlib.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strdup, (const char *src)) {
  if (src == nullptr) {
    return nullptr;
  }
  size_t len = internal::string_length(src) + 1;
  char *dest = reinterpret_cast<char *>(::malloc(len));
  if (dest == nullptr) {
    return nullptr;
  }
  inline_memcpy(dest, src, len);
  return dest;
}

} // namespace __llvm_libc
