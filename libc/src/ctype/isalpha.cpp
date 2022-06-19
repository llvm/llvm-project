//===-- Implementation of isalpha------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isalpha.h"

#include "src/__support/common.h"
#include "src/__support/ctype_utils.h"

namespace __llvm_libc {

// TODO: Currently restricted to default locale.
// These should be extended using locale information.
LLVM_LIBC_FUNCTION(int, isalpha, (int c)) {
  return static_cast<int>(internal::isalpha(static_cast<unsigned>(c)));
}

} // namespace __llvm_libc
