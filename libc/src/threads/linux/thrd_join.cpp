//===-- Linux implementation of the thrd_join function --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_join.h"
#include "config/linux/syscall.h" // For syscall function.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "include/threads.h"      // For thrd_* type definitions.
#include "src/__support/common.h"
#include "src/sys/mman/munmap.h"
#include "src/threads/linux/Futex.h"
#include "src/threads/linux/Thread.h"

#include <linux/futex.h> // For futex operations.
#include <stdatomic.h>   // For atomic_load.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, thrd_join, (thrd_t * thread, int *retval)) {
  FutexWord *clear_tid_address =
      reinterpret_cast<FutexWord *>(thread->__clear_tid);

  // The kernel should set the value at the clear tid address to zero.
  // If not, it is a spurious wake and we should continue to wait on
  // the futex.
  while (atomic_load(clear_tid_address) != 0) {
    // We cannot do a FUTEX_WAIT_PRIVATE here as the kernel does a
    // FUTEX_WAKE and not a FUTEX_WAKE_PRIVATE.
    __llvm_libc::syscall(SYS_futex, clear_tid_address, FUTEX_WAIT,
                         ThreadParams::ClearTIDValue, nullptr);
  }

  *retval = thread->__retval;

  if (__llvm_libc::munmap(thread->__stack, thread->__stack_size) == -1)
    return thrd_error;

  return thrd_success;
}

} // namespace __llvm_libc
