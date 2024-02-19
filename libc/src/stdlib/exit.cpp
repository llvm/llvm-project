//===-- Implementation of exit --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/exit.h"
#include "src/__support/OSUtil/quick_exit.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

namespace internal {
void call_exit_callbacks();
}

LLVM_LIBC_FUNCTION(void, exit, (int status)) {
  internal::call_exit_callbacks();
  quick_exit(status);
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE
