//===-- Implementation of strrchr------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/strrchr.h"

#include "src/__support/common.h"
#include "src/string/string_utils.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, strrchr, (const char *src, int c)) {
  return internal::strrchr_implementation(src, c);
}

} // namespace __llvm_libc
