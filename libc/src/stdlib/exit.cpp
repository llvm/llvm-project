//===-- Implementation of exit --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/exit.h"
#include "src/__support/common.h"
#include "src/stdlib/_Exit.h"

namespace __llvm_libc {

namespace internal {
void call_exit_callbacks();
}

LLVM_LIBC_FUNCTION(void, exit, (int status)) {
  internal::call_exit_callbacks();
  _Exit(status);
}

} // namespace __llvm_libc
