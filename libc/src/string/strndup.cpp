//===-- Implementation of strndup -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strndup.h"
#include "src/string/memcpy.h"
#include "src/string/string_utils.h"

#include "src/__support/common.h"

#include <stddef.h>
#include <stdlib.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strndup, (const char *src, size_t size)) {
  if (src == nullptr)
    return nullptr;
  size_t len = internal::string_length(src);
  if (len > size)
    len = size;
  char *dest = reinterpret_cast<char *>(::malloc(len + 1)); // NOLINT
  if (dest == nullptr)
    return nullptr;
  char *result =
      reinterpret_cast<char *>(__llvm_libc::memcpy(dest, src, len + 1));
  result[len] = '\0';
  return result;
}

} // namespace __llvm_libc
