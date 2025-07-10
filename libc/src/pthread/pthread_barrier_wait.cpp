//===-- Linux implementation of the pthread_barrier_init function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutex_init.h"
#include "pthread_mutexattr.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/barrier.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_barrier_wait,
                   (pthread_barrier_t * b,
                    const pthread_barrierattr_t *__restrict attr,
                    unsigned count)) {
  return reinterpret_cast<Barrier *>(b)->wait();
}

} // namespace LIBC_NAMESPACE_DECL
