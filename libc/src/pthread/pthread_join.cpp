//===-- Implementation of the pthread_join function -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_join.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <pthread.h> // For pthread_* type definitions.

namespace LIBC_NAMESPACE {

static_assert(sizeof(pthread_t) == sizeof(LIBC_NAMESPACE::Thread),
              "Mismatch between pthread_t and internal Thread.");

LLVM_LIBC_FUNCTION(int, pthread_join, (pthread_t th, void **retval)) {
  auto *thread = reinterpret_cast<Thread *>(&th);
  int result = thread->join(retval);
  return result;
}

} // namespace LIBC_NAMESPACE
