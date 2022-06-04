//===-- Implementation of the pthread_detach function ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_detach.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <pthread.h> // For pthread_* type definitions.

namespace __llvm_libc {

static_assert(sizeof(pthread_t) == sizeof(__llvm_libc::Thread<void *>),
              "Mismatch between pthread_t and internal Thread<void *>.");

LLVM_LIBC_FUNCTION(int, pthread_detach, (pthread_t th)) {
  auto *thread = reinterpret_cast<Thread<void *> *>(&th);
  thread->detach();
  return 0;
}

} // namespace __llvm_libc
