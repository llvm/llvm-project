//===-- Implementation of pthread_spin_init function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_spin_init.h"
#include "hdr/errno_macros.h"
#include "src/__support/CPP/new.h"
#include "src/__support/common.h"
#include "src/__support/threads/spin_lock.h"
#include <pthread.h> // for PTHREAD_PROCESS_SHARED, PTHREAD_PROCESS_PRIVATE

namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_spinlock_t::__lockword) == sizeof(SpinLock) &&
                  alignof(decltype(pthread_spinlock_t::__lockword)) ==
                      alignof(SpinLock),
              "pthread_spinlock_t::__lockword and SpinLock must be of the same "
              "size and alignment");

LLVM_LIBC_FUNCTION(int, pthread_spin_init,
                   (pthread_spinlock_t * lock, [[maybe_unused]] int pshared)) {
  if (!lock)
    return EINVAL;
  if (pshared != PTHREAD_PROCESS_SHARED && pshared != PTHREAD_PROCESS_PRIVATE)
    return EINVAL;
  // The spin lock here is a simple atomic flag, so we don't need to do any
  // special handling for pshared.
  ::new (&lock->__lockword) SpinLock();
  lock->__owner = 0;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
