//===-- Linux implementation of the pthread_mutex_trylock function --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "pthread_mutex_trylock.h"

#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"
#include "src/__support/threads/mutex.h"
#include "src/__support/threads/mutex_common.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(int, pthread_mutex_trylock, (pthread_mutex_t * mutex)) {
  MutexError err = reinterpret_cast<Mutex *>(mutex)->try_lock();
  if (err == MutexError::DEADLOCK)
    return EDEADLK;
  if (err == MutexError::BUSY)
    return EBUSY;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
