//===-- Linux implementation of the thrd_create function ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Futex.h"

#include "src/__support/OSUtil/syscall.h" // For syscall function.
#include "src/__support/architectures.h"
#include "src/__support/common.h"
#include "src/threads/linux/Thread.h"
#include "src/threads/thrd_create.h"

#include <errno.h>       // For E* error values.
#include <linux/sched.h> // For CLONE_* flags.
#include <stdint.h>
#include <sys/mman.h>    // For PROT_* and MAP_* definitions.
#include <sys/syscall.h> // For syscall numbers.
#include <threads.h>     // For thrd_* type definitions.

#ifdef SYS_mmap2
constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap2;
#elif SYS_mmap
constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap;
#else
#error "SYS_mmap or SYS_mmap2 not available on the target platform"
#endif

namespace __llvm_libc {

// We align the start args to 16-byte boundary as we adjust the allocated
// stack memory with its size. We want the adjusted address to be at a
// 16-byte boundary to satisfy the x86_64 and aarch64 ABI requirements.
// If different architecture in future requires higher alignment, then we
// can add a platform specific alignment spec.
struct alignas(16) StartArgs {
  thrd_t *thread;
  thrd_start_t func;
  void *arg;
};

__attribute__((always_inline)) inline uintptr_t get_start_args_addr() {
  // NOTE: For __builtin_frame_address to work reliably across compilers,
  // architectures and various optimization levels, the TU including this file
  // should be compiled with -fno-omit-frame-pointer.
  return reinterpret_cast<uintptr_t>(__builtin_frame_address(0))
         // The x86_64 call instruction pushes resume address on to the stack.
         // Next, The x86_64 SysV ABI requires that the frame pointer be pushed
         // on to the stack. Similarly on aarch64, previous frame pointer and
         // the value of the link register are pushed on to the stack. So, in
         // both these cases, we have to step past two 64-bit values to get
         // to the start args.
         + sizeof(uintptr_t) * 2;
}

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

  // Allocate thread stack.
  long mmap_result =
      __llvm_libc::syscall(MMAP_SYSCALL_NUMBER,
                           0, // No special address
                           ThreadParams::DEFAULT_STACK_SIZE,
                           PROT_READ | PROT_WRITE,      // Read and write stack
                           MAP_ANONYMOUS | MAP_PRIVATE, // Process private
                           -1, // Not backed by any file
                           0   // No offset
      );
  if (mmap_result < 0 && (uintptr_t(mmap_result) >=
                          UINTPTR_MAX - ThreadParams::DEFAULT_STACK_SIZE)) {
    return -mmap_result == ENOMEM ? thrd_nomem : thrd_error;
  }
  void *stack = reinterpret_cast<void *>(mmap_result);

  thread->__stack = stack;
  thread->__stack_size = ThreadParams::DEFAULT_STACK_SIZE;
  thread->__retval = -1;
  FutexWordType *clear_tid_address = &thread->__clear_tid.__word;
  *clear_tid_address = ThreadParams::CLEAR_TID_VALUE;

  // When the new thread is spawned by the kernel, the new thread gets the
  // stack we pass to the clone syscall. However, this stack is empty and does
  // not have any local vars present in this function. Hence, one cannot
  // pass arguments to the thread start function, or use any local vars from
  // here. So, we pack them into the new stack from where the thread can sniff
  // them out.
  uintptr_t adjusted_stack = reinterpret_cast<uintptr_t>(stack) +
                             ThreadParams::DEFAULT_STACK_SIZE -
                             sizeof(StartArgs);
  StartArgs *start_args = reinterpret_cast<StartArgs *>(adjusted_stack);
  start_args->thread = thread;
  start_args->func = func;
  start_args->arg = arg;

  // The clone syscall takes arguments in an architecture specific order.
  // Also, we want the result of the syscall to be in a register as the child
  // thread gets a completely different stack after it is created. The stack
  // variables from this function will not be availalbe to the child thread.
#ifdef LLVM_LIBC_ARCH_X86_64
  long register clone_result asm("rax");
  clone_result = __llvm_libc::syscall(
      SYS_clone, clone_flags, adjusted_stack,
      &thread->__tid,    // The address where the child tid is written
      clear_tid_address, // The futex where the child thread status is signalled
      0                  // Set TLS to null for now.
  );
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  long register clone_result asm("x0");
  clone_result = __llvm_libc::syscall(
      SYS_clone, clone_flags, adjusted_stack,
      &thread->__tid,   // The address where the child tid is written
      0,                // Set TLS to null for now.
      clear_tid_address // The futex where the child thread status is signalled
  );
#else
#error "Unsupported architecture for the clone syscall."
#endif

  if (clone_result == 0) {
    start_thread();
  } else if (clone_result < 0) {
    __llvm_libc::syscall(SYS_munmap, mmap_result,
                         ThreadParams::DEFAULT_STACK_SIZE);
    int error_val = -clone_result;
    return error_val == ENOMEM ? thrd_nomem : thrd_error;
  }

  return thrd_success;
}

} // namespace __llvm_libc
