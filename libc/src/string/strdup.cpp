//===-- Implementation of strdup ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strdup.h"
#include "src/errno/libc_errno.h"
#include "src/string/allocating_string_utils.h"
#include "src/string/memory_utils/inline_memcpy.h"

#include "src/__support/common.h"

#include <stdlib.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strdup, (const char *src)) {
  auto dup = internal::strdup(src);
  if (dup)
    return *dup;
  if (src != nullptr)
    libc_errno = ENOMEM;
  return nullptr;
}

} // namespace __llvm_libc
