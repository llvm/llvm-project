//===-- Implementation of the pthread_attr_init ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_attr_init.h"

#include "src/__support/common.h"
#include "src/__support/threads/thread.h" // For thread::DEFAULT_*

#include <pthread.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, pthread_attr_init, (pthread_attr_t * attr)) {
  *attr = pthread_attr_t{
      PTHREAD_CREATE_JOINABLE,   // Not detached
      nullptr,                   // Let the thread manage its stack
      Thread::DEFAULT_STACKSIZE, // stack size.
      Thread::DEFAULT_GUARDSIZE, // Default page size for the guard size.
  };
  return 0;
}

} // namespace __llvm_libc
