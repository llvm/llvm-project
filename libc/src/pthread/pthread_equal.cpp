//===-- Implementation of the pthread_equal function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_equal.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/thread.h"

#include <pthread.h> // For pthread_* type definitions.

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_equal, (pthread_t lhs, pthread_t rhs)) {
  auto *lhs_internal = reinterpret_cast<Thread *>(&lhs);
  auto *rhs_internal = reinterpret_cast<Thread *>(&rhs);
  return *lhs_internal == *rhs_internal;
}

} // namespace LIBC_NAMESPACE_DECL
