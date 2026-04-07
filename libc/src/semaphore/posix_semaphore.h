//===-- Shared helpers for POSIX semaphores -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEMAPHORE_POSIX_SEMAPHORE_H
#define LLVM_LIBC_SRC_SEMAPHORE_POSIX_SEMAPHORE_H

#include "hdr/types/sem_t.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/common.h"
#include "src/__support/threads/linux/futex_word.h"

namespace LIBC_NAMESPACE_DECL {
namespace sem_utils {

// 0x53 = S, 0x45 = E, 0x4D = M, 0x31 = 1
// canary value: SEM1 in hex
LIBC_INLINE_VAR constexpr unsigned int SEM_CANARY = 0x53454D31U;

static_assert(sizeof(__futex_word) == sizeof(FutexWordType));
static_assert(alignof(__futex_word) >= alignof(FutexWordType));

// TODO:
// 1. Add named semaphore support: sem_open, sem_close, sem_unlink
// 2. Add the posting and waiting operations: sem_post, sem_wait,
//    sem_trywait, sem_timedwait, sem_clockwait.

LIBC_INLINE FutexWordType *value_ptr(sem_t *sem) {
  return &sem->__value.__word;
}

// get the atomic reference for sem->__value
LIBC_INLINE cpp::AtomicRef<FutexWordType> value(sem_t *sem) {
  return cpp::AtomicRef<FutexWordType>(*value_ptr(sem));
}

LIBC_INLINE void initialize(sem_t *sem, unsigned int initial_value) {
  // used in sem_init
  // init happens before the semaphore is published to any threads
  // initialize a initialized semaphore is undefined
  // so RELAXED ordering is enough
  value(sem).store(initial_value, cpp::MemoryOrder::RELAXED);
  sem->__canary = SEM_CANARY;
  for (unsigned char &byte : sem->__reserved)
    byte = 0;
}

LIBC_INLINE bool is_valid(const sem_t *sem) {
  // sanity check for a given semaphore pointer
  return sem != nullptr && sem->__canary == SEM_CANARY;
}

LIBC_INLINE void invalidate(sem_t *sem) {
  // used in sem_destroy
  // invalidate is safe only when no threads is using
  // blocked by destroyed semaphore is undefined
  // use a destroyed semaphore is undefined
  // RELAXED ordering is enough
  value(sem).store(0, cpp::MemoryOrder::RELAXED);
  sem->__canary = 0;
  for (unsigned char &byte : sem->__reserved)
    byte = 0;
}

} // namespace sem_utils
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_POSIX_SEMAPHORE_H
