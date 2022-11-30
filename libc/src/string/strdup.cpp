//===-- Implementation of strdup ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strdup.h"
#include "src/string/allocating_string_utils.h"
#include "src/string/memory_utils/memcpy_implementations.h"

#include "src/__support/common.h"

#include <stdlib.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strdup, (const char *src)) {
  return internal::strdup(src);
}

} // namespace __llvm_libc
