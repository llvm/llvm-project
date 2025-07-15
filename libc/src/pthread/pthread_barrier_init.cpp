//===-- Linux implementation of the pthread_barrier_init function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_barrier_init.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/barrier.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

static_assert(
    sizeof(Barrier) <= sizeof(pthread_barrier_t),
    "The public pthread_barrier_t type cannot accommodate the internal "
    "barrier type.");

LLVM_LIBC_FUNCTION(int, pthread_barrier_init,
                   (pthread_barrier_t * b,
                    const pthread_barrierattr_t *__restrict attr,
                    unsigned count)) {
  return Barrier::init(reinterpret_cast<Barrier *>(b), attr, count);
}

} // namespace LIBC_NAMESPACE_DECL
