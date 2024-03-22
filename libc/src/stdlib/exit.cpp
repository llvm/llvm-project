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

extern "C" void __cxa_finalize(void *);

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(void, exit, (int status)) {
  __cxa_finalize(nullptr);
  quick_exit(status);
  __builtin_unreachable();
}

} // namespace LIBC_NAMESPACE
