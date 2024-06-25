//===-- Implementation for Rwlock's destroy function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_rwlock_destroy.h"

#include "src/__support/common.h"
#include "src/__support/threads/linux/rwlock.h"

#include <errno.h>
#include <pthread.h>

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(int, pthread_rwlock_destroy, (pthread_rwlock_t * rwlock)) {
  if (!rwlock)
    return EINVAL;
  auto *rw = reinterpret_cast<RwLock *>(rwlock);
  RwLock::LockResult res = rw->check_for_destroy();

  // this is currently no-op, but we still call the destructor as a symmetry
  // to its constructor call;
  if (res == RwLock::LockResult::Success)
    rw->~RwLock();

  return static_cast<int>(res);
}

} // namespace LIBC_NAMESPACE
