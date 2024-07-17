//===-- Linux implementation of the thrd_create function ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_create.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/thread.h"

#include <errno.h>
#include <threads.h> // For thrd_* type definitions.

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(thrd_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between thrd_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, thrd_create,
                   (thrd_t * th, thrd_start_t func, void *arg)) {
  auto *thread = reinterpret_cast<LIBC_NAMESPACE::Thread *>(th);
  int result = thread->run(func, arg);
  if (result == 0)
    return thrd_success;
  else if (result == ENOMEM)
    return thrd_nomem;
  else
    return thrd_error;
}

} // namespace LIBC_NAMESPACE_DECL
