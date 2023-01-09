//===-- Implementation of strncmp -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncmp.h"

#include "src/__support/CPP/functional.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/strcmp_implementations.h"

#include <stddef.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, strncmp,
                   (const char *left, const char *right, size_t n)) {
  return strncmp_implementation(left, right, n, cpp::minus<char>{});
}

} // namespace __llvm_libc
