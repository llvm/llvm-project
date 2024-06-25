//===-- Implementation of exit --------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/exit.h"
#include "src/__support/OSUtil/exit.h"
#include "src/__support/common.h"

extern "C" void __cxa_finalize(void *);

namespace LIBC_NAMESPACE {

[[noreturn]] LLVM_LIBC_FUNCTION(void, exit, (int status)) {
  __cxa_finalize(nullptr);
  internal::exit(status);
}

} // namespace LIBC_NAMESPACE
