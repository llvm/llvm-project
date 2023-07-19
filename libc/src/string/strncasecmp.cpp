//===-- Implementation of strncasecmp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strncasecmp.h"

#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"
#include "src/string/memory_utils/inline_strcmp.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, strncasecmp,
                   (const char *left, const char *right, size_t n)) {
  auto case_cmp = [](char a, char b) {
    return __llvm_libc::internal::tolower(a) -
           __llvm_libc::internal::tolower(b);
  };
  return inline_strncmp(left, right, n, case_cmp);
}

} // namespace __llvm_libc
