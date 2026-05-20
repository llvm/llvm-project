//===-- Linux Semaphore implementation for POSIX semaphores ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SEMAPHORE_LINUX_SEMAPHORE_H
#define LLVM_LIBC_SRC_SEMAPHORE_LINUX_SEMAPHORE_H

#include "hdr/types/mode_t.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/common.h"
#include "src/__support/error_or.h"
#include "src/__support/threads/futex_utils.h"

namespace LIBC_NAMESPACE_DECL {

class Semaphore {
  Futex value;
  unsigned int canary;

  // A private constant canary used to detect use of uninitialized or
  // destroyed semaphores. Chose "SEM1" in ASCII (0x53='S', 0x45='E',
  // 0x4D='M', 0x31='1').
  static constexpr unsigned int SEM_CANARY = 0x53454D31U;

public:
  // TODO:
  // Add the posting and waiting operations: sem_post, sem_wait,
  //    sem_trywait, sem_timedwait, sem_clockwait.

  LIBC_INLINE constexpr Semaphore(unsigned int value)
      : value(value), canary(SEM_CANARY) {}

  // Sanity check to detect use of uninitialized or destroyed semaphores.
  LIBC_INLINE bool is_valid() const { return canary == SEM_CANARY; }

  LIBC_INLINE void destroy() {
    // Destroying a semaphore while threads are blocked on it is undefined
    // behavior. Similarly, using a destroyed semaphore is undefined.
    // Therefore no concurrency safe destruction is required here,
    // RELAXED memory ordering is sufficient.
    value.store(0, cpp::MemoryOrder::RELAXED);
    canary = 0;
  }

  LIBC_INLINE int getvalue() const {
    // get value is informational, not a synchronization op.
    // RELAXED ordering is enough.
    // TODO: handle the case where the semaphore is locked.
    return static_cast<int>(
        const_cast<Futex &>(value).load(cpp::MemoryOrder::RELAXED));
  }

  // Named semaphore operations.
  // creates or opens a named semaphore backed by a file in /dev/shm/.
  // When O_CREAT is specified in oflag, mode and value are used for
  // initialization.
  static ErrorOr<Semaphore *> open(const char *name, int oflag, mode_t mode,
                                   unsigned int value);

  // unmaps a named semaphore.
  static int close(Semaphore *sem);

  // removes a named semaphore from the filesystem.
  static int unlink(const char *name);
};

} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC_SEMAPHORE_LINUX_SEMAPHORE_H
