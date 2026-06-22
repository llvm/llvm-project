//===-- Linux implementation of the pthread_mutex_lock function -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutex_lock.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_mutex_lock, (pthread_mutex_t * mutex)) {
  MutexError err = reinterpret_cast<Mutex *>(mutex)->lock();
  if (err == MutexError::DEADLOCK)
    return EDEADLK;
  // TODO: When the Mutex class supports all the possible error conditions
  // return the appropriate error value here.
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
