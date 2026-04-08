//===-- Internal Semaphore types and helpers for POSIX semaphores ---------===//
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
#include "src/__support/threads/linux/futex_utils.h"

namespace LIBC_NAMESPACE_DECL {

class Semaphore {
  Futex value;
  unsigned int canary;

  // 0x53 = S, 0x45 = E, 0x4D = M, 0x31 = 1
  static constexpr unsigned int SEM_CANARY = 0x53454D31U;

public:
  // TODO:
  // 1. Add named semaphore support: sem_open, sem_close, sem_unlink
  // 2. Add the posting and waiting operations: sem_post, sem_wait,
  //    sem_trywait, sem_timedwait, sem_clockwait.

  LIBC_INLINE void init(unsigned int initial_value) {
    // init happens before the semaphore is published to any threads
    // initialize a initialized semaphore is undefined
    // so RELAXED ordering is enough
    value.store(initial_value, cpp::MemoryOrder::RELAXED);
    canary = SEM_CANARY;
  }

  // sanity check for a given semaphore pointer
  LIBC_INLINE bool is_valid() const { return canary == SEM_CANARY; }

  LIBC_INLINE void destroy() {
    // invalidate is safe only when no threads is using
    // blocked by destroyed semaphore is undefined
    // use a destroyed semaphore is undefined
    // RELAXED ordering is enough
    value.store(0, cpp::MemoryOrder::RELAXED);
    canary = 0;
  }

  LIBC_INLINE int getvalue() const {
    // get value is informational but not a synchronization op
    // RELAXED ordering is enough
    return static_cast<int>(
        const_cast<Futex &>(value).load(cpp::MemoryOrder::RELAXED));
  }
};

static_assert(sizeof(Semaphore) <= sizeof(sem_t),
              "Semaphore must fit within sem_t.");
static_assert(alignof(Semaphore) <= alignof(sem_t),
              "Semaphore alignment must be compatible with sem_t.");

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_POSIX_SEMAPHORE_H
