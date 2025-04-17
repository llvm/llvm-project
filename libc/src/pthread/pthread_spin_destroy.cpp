//===-- Implementation of pthread_spin_destroy function -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/pthread/pthread_spin_destroy.h"
#include "hdr/errno_macros.h"
#include "src/__support/common.h"
#include "src/__support/threads/spin_lock.h"
namespace LIBC_NAMESPACE_DECL {

static_assert(sizeof(pthread_spinlock_t::__lockword) == sizeof(SpinLock) &&
                  alignof(decltype(pthread_spinlock_t::__lockword)) ==
                      alignof(SpinLock),
              "pthread_spinlock_t::__lockword and SpinLock must be of the same "
              "size and alignment");

LLVM_LIBC_FUNCTION(int, pthread_spin_destroy,
                   ([[maybe_unused]] pthread_spinlock_t * lock)) {
  // If an implementation detects that the value specified by the lock argument
  // to pthread_spin_lock() or pthread_spin_trylock() does not refer to an
  // initialized spin lock object, it is recommended that the function should
  // fail and report an [EINVAL] error.
  if (!lock)
    return EINVAL;
  auto spin_lock = reinterpret_cast<SpinLock *>(&lock->__lockword);
  if (!spin_lock || spin_lock->is_invalid())
    return EINVAL;

  // If an implementation detects that the value specified by the lock argument
  // to pthread_spin_destroy() or pthread_spin_init() refers to a locked spin
  // lock object, or detects that the value specified by the lock argument to
  // pthread_spin_init() refers to an already initialized spin lock object, it
  // is recommended that the function should fail and report an [EBUSY] error.
  if (spin_lock->is_locked())
    return EBUSY;

  // poison the lock
  spin_lock->~SpinLock();
  lock->__owner = 0;
  return 0;
}

} // namespace LIBC_NAMESPACE_DECL
