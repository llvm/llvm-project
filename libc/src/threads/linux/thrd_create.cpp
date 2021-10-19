//===-- Linux implementation of the thrd_create function ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/threads/thrd_create.h"
#include "config/linux/syscall.h" // For syscall function.
#include "include/errno.h"        // For E* error values.
#include "include/sys/mman.h"     // For PROT_* and MAP_* definitions.
#include "include/sys/syscall.h"  // For syscall numbers.
#include "include/threads.h"      // For thrd_* type definitions.
#include "src/__support/common.h"
#include "src/errno/llvmlibc_errno.h"
#include "src/sys/mman/mmap.h"
#include "src/sys/mman/munmap.h"
#include "src/threads/linux/Futex.h"
#include "src/threads/linux/Thread.h"

#include <linux/sched.h> // For CLONE_* flags.
#include <stdint.h>

namespace __llvm_libc {

struct StartArgs {
  thrd_t *thread;
  thrd_start_t func;
  void *arg;
};

static __attribute__((noinline)) void start_thread() {
  StartArgs *start_args = reinterpret_cast<StartArgs *>(get_start_args_addr());
  __llvm_libc::syscall(SYS_exit, start_args->thread->__retval =
                                     start_args->func(start_args->arg));
}

LLVM_LIBC_FUNCTION(int, thrd_create,
                   (thrd_t * thread, thrd_start_t func, void *arg)) {
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
  FutexWord *clear_tid_address =
      reinterpret_cast<FutexWord *>(thread->__clear_tid);
  *clear_tid_address = ThreadParams::ClearTIDValue;

  // When the new thread is spawned by the kernel, the new thread gets the
  // stack we pass to the clone syscall. However, this stack is empty and does
  // not have any local vars present in this function. Hence, one cannot
  // pass arguments to the thread start function, or use any local vars from
  // here. So, we pack them into the new stack from where the thread can sniff
  // them out.
  uintptr_t adjusted_stack = reinterpret_cast<uintptr_t>(stack) +
                             ThreadParams::DefaultStackSize - sizeof(StartArgs);
  StartArgs *start_args = reinterpret_cast<StartArgs *>(adjusted_stack);
  start_args->thread = thread;
  start_args->func = func;
  start_args->arg = arg;

  // TODO: The arguments to the clone syscall below is correct for x86_64
  // but it might differ for other architectures. So, make this call
  // architecture independent. May be implement a glibc like wrapper for clone
  // and use it here.
  long register clone_result asm("rax");
  clone_result =
      __llvm_libc::syscall(SYS_clone, clone_flags, adjusted_stack,
                           &thread->__tid, clear_tid_address, 0);

  if (clone_result == 0) {
    start_thread();
  } else if (clone_result < 0) {
    __llvm_libc::munmap(thread->__stack, thread->__stack_size);
    int error_val = -clone_result;
    return error_val == ENOMEM ? thrd_nomem : thrd_error;
  }

  return thrd_success;
}

} // namespace __llvm_libc
