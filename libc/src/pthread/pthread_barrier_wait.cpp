//===-- Linux implementation of the pthread_barrier_init function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_barrier_wait.h"

#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/barrier.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_barrier_wait, (pthread_barrier_t * b)) {
  int out = reinterpret_cast<Barrier *>(b)->wait();
  if (out == BARRIER_FIRST_EXITED)
    return PTHREAD_BARRIER_SERIAL_THREAD;
  
  return out;
}

} // namespace LIBC_NAMESPACE_DECL
