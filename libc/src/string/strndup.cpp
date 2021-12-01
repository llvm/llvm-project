//===-- Implementation of strndup -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strndup.h"
#include "src/string/memory_utils/memcpy_implementations.h"
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
  char *dest = reinterpret_cast<char *>(::malloc(len + 1));
  if (dest == nullptr)
    return nullptr;
  inline_memcpy(dest, src, len + 1);
  dest[len] = '\0';
  return dest;
}

} // namespace __llvm_libc
