//===-- Implementation of isascii------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/ctype/isascii.h"

#include "src/__support/common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, isascii, (int c)) {
  return static_cast<int>((c & (~0x7f)) == 0);
}

} // namespace __llvm_libc
