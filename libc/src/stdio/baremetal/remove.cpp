//===-- Linux implementation of remove ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/remove.h"

#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

// TODO: See https://github.com/llvm/llvm-project/issues/85335 for more details
// on why this is needed.

LLVM_LIBC_FUNCTION(int, remove, (const char *)) {
  return -1;
}

} // namespace LIBC_NAMESPACE
