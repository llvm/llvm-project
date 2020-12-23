//===-- Implementation of isxdigit-----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isxdigit.h"
#include "src/ctype/ctype_utils.h"

#include "src/__support/common.h"

namespace __llvm_libc {

// TODO: Currently restricted to default locale.
// These should be extended using locale information.
LLVM_LIBC_FUNCTION(int, isxdigit, (int c)) {
  const unsigned ch = c;
  return internal::isdigit(ch) || (ch | 32) - 'a' < 6;
}

} // namespace __llvm_libc
