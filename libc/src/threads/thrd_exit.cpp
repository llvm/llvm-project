//===-- Linux implementation of the thrd_exit function --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_exit.h"
#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <threads.h> // For thrd_* type definitions.

namespace LIBC_NAMESPACE {

static_assert(sizeof(thrd_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between thrd_t and internal Thread.");

LLVM_LIBC_FUNCTION(void, thrd_exit, (int retval)) {
  thread_exit(ThreadReturnValue(retval), ThreadStyle::STDC);
}

} // namespace LIBC_NAMESPACE
