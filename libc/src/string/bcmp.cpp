//===-- Implementation of bcmp --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/string/bcmp.h"
#include "src/__support/common.h"
#include "src/string/memory_utils/inline_bcmp.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, bcmp,
                   (const void *lhs, const void *rhs, size_t count)) {
  return inline_bcmp(lhs, rhs, count);
}

} // namespace __llvm_libc
