//===-- Implementation of index -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/index.h"

#include "src/__support/common.h"
#include "src/string/memory_utils/strchr_implementations.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(char *, index, (const char *src, int c)) {
  return strchr_implementation(src, c);
}

} // namespace __llvm_libc
