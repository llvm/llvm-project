//===-- Linux implementation of the pthread_create function ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_create.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h"

#include <errno.h>
#include <pthread.h> // For pthread_* type definitions.

namespace __llvm_libc {

static_assert(sizeof(pthread_t) == sizeof(__llvm_libc::Thread<int>),
              "Mismatch between pthread_t and internal Thread<int>.");

LLVM_LIBC_FUNCTION(int, pthread_create,
                   (pthread_t *__restrict th,
                    const pthread_attr_t *__restrict attr,
                    __pthread_start_t func, void *arg)) {
  auto *thread = reinterpret_cast<__llvm_libc::Thread<void *> *>(th);
  int result = thread->run(func, arg, nullptr, 0);
  if (result != 0 && result != EPERM)
    return EAGAIN;
  return result;
}

} // namespace __llvm_libc
