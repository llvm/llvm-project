//===-- Linux implementation of the pthread_mutex_unlock function ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutex_unlock.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/libc_assert.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"

#include <pthread.h>

namespace LIBC_NAMESPACE_DECL {

// The implementation currently handles only plain mutexes.
LLVM_LIBC_FUNCTION(int, pthread_mutex_unlock, (pthread_mutex_t * mutex)) {
  MutexError err = reinterpret_cast<Mutex *>(mutex)->unlock();
  // Per-POSIX specification and our implementation, EPERM is the only possible
  // error.
  if (err == MutexError::UNLOCK_WITHOUT_LOCK)
    return EPERM;
  LIBC_ASSERT(err == MutexError::NONE);
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
