//===-- Implementation of strdup ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strdup.h"
#include "src/string/memcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

#include <stdlib.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strdup, (const char *src)) {
  if (src == nullptr) {
    return nullptr;
  }
  size_t len = internal::string_length(src) + 1;
  char *dest = reinterpret_cast<char *>(::malloc(len)); // NOLINT
  if (dest == nullptr) {
    return nullptr;
  }
  char *result = reinterpret_cast<char *>(__llvm_libc::memcpy(dest, src, len));
  return result;
}

} // namespace __llvm_libc
