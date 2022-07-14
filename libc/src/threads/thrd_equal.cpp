//===-- Implementation of the thrd_equal function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_equal.h"
#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <threads.h> // For thrd_* type definitions.

namespace __llvm_libc {

static_assert(sizeof(thrd_t) == sizeof(__llvm_libc::Thread),
              "Mismatch between thrd_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, thrd_equal, (thrd_t lhs, thrd_t rhs)) {
  auto *lhs_internal = reinterpret_cast<Thread *>(&lhs);
  auto *rhs_internal = reinterpret_cast<Thread *>(&rhs);
  return *lhs_internal == *rhs_internal;
}

} // namespace __llvm_libc
