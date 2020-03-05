//===---------- Linux implementation of the thrd_create function ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "config/linux/syscall.h" // For syscall function.
#include "include/errno.h"        // For E* error values.
#include "include/sys/mman.h"     // For PROT_* and MAP_* definitions.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "include/threads.h"      // For thrd_* type definitions.
#include "src/__support/common.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/threads/linux/thread_utils.h"

#include <linux/futex.h> // For futex operations.
#include <linux/sched.h> // For CLONE_* flags.
#include <stdint.h>

namespace __llvm_libc {

static void start_thread(thrd_t *thread, thrd_start_t func, void *arg) {
  __llvm_libc::syscall(SYS_exit, thread->__retval = func(arg));
}

int LLVM_LIBC_ENTRYPOINT(thrd_create)(thrd_t *thread, thrd_start_t func,
                                      void *arg) {
  unsigned clone_flags =
      CLONE_VM        // Share the memory space with the parent.
      | CLONE_FS      // Share the file system with the parent.
      | CLONE_FILES   // Share the files with the parent.
      | CLONE_SIGHAND // Share the signal handlers with the parent.
      | CLONE_THREAD  // Same thread group as the parent.
      | CLONE_SYSVSEM // Share a single list of System V semaphore adjustment
                      // values
      | CLONE_PARENT_SETTID   // Set child thread ID in |ptid| of the parent.
      | CLONE_CHILD_CLEARTID; // Let the kernel clear the tid address and futex
                              // wake the joining thread.
  // TODO: Add the CLONE_SETTLS flag and setup the TLS area correctly when
  // making the clone syscall.

  void *stack = __llvm_libc::mmap(nullptr, ThreadParams::DefaultStackSize,
                                  PROT_READ | PROT_WRITE,
                                  MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  if (stack == MAP_FAILED)
    return llvmlibc_errno == ENOMEM ? thrd_nomem : thrd_error;

  thread->__stack = stack;
  thread->__stack_size = ThreadParams::DefaultStackSize;
  thread->__retval = -1;
  FutexData *clear_tid_address =
      reinterpret_cast<FutexData *>(thread->__clear_tid);
  *clear_tid_address = ThreadParams::ClearTIDValue;

  long clone_result = __llvm_libc::syscall(
      SYS_clone, clone_flags,
      reinterpret_cast<uintptr_t>(stack) + ThreadParams::DefaultStackSize - 1,
      &thread->__tid, clear_tid_address, 0);

  if (clone_result == 0) {
    start_thread(thread, func, arg);
  } else if (clone_result < 0) {
    int error_val = -clone_result;
    return error_val == ENOMEM ? thrd_nomem : thrd_error;
  }

  return thrd_success;
}

} // namespace __llvm_libc
