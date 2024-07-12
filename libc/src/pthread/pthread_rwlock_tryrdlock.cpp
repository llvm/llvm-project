//===-- Implementation of the Rwlock's tryrdlock function -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_rwlock_tryrdlock.h"

#include "src/__support/common.h"
#include "src/__support/threads/linux/rwlock.h"

#include <errno.h>
#include <pthread.h>

namespace LIBC_NAMESPACE {

static_assert(
    sizeof(RwLock) == sizeof(pthread_rwlock_t) &&
        alignof(RwLock) == alignof(pthread_rwlock_t),
    "The public pthread_rwlock_t type must be of the same size and alignment "
    "as the internal rwlock type.");

LLVM_LIBC_FUNCTION(int, pthread_rwlock_tryrdlock, (pthread_rwlock_t * rwlock)) {
  if (!rwlock)
    return EINVAL;
  RwLock *rw = reinterpret_cast<RwLock *>(rwlock);
  return static_cast<int>(rw->try_read_lock());
}

} // namespace LIBC_NAMESPACE
