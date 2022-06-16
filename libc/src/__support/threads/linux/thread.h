//===--- Implementation of a Linux thread class -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_THREAD_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_THREAD_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/CPP/error.h"
#include "src/__support/OSUtil/syscall.h"           // For syscall functions.
#include "src/__support/threads/linux/futex_word.h" // For FutexWordType
#include "src/__support/threads/thread_attrib.h"

#ifdef LLVM_LIBC_ARCH_AARCH64
#include <arm_acle.h>
#endif

#include <linux/futex.h>
#include <linux/sched.h> // For CLONE_* flags.
#include <stdint.h>
#include <sys/mman.h>    // For PROT_* and MAP_* definitions.
#include <sys/syscall.h> // For syscall numbers.

namespace __llvm_libc {

template <typename ReturnType> struct Thread;

#ifdef SYS_mmap2
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap2;
#elif SYS_mmap
static constexpr long MMAP_SYSCALL_NUMBER = SYS_mmap;
#else
#error "SYS_mmap or SYS_mmap2 not available on the target platform"
#endif

static constexpr size_t DEFAULT_STACK_SIZE = (1 << 16); // 64KB
static constexpr uint32_t CLEAR_TID_VALUE = 0xABCD1234;
static constexpr unsigned CLONE_SYSCALL_FLAGS =
    CLONE_VM        // Share the memory space with the parent.
    | CLONE_FS      // Share the file system with the parent.
    | CLONE_FILES   // Share the files with the parent.
    | CLONE_SIGHAND // Share the signal handlers with the parent.
    | CLONE_THREAD  // Same thread group as the parent.
    | CLONE_SYSVSEM // Share a single list of System V semaphore adjustment
                    // values
    | CLONE_PARENT_SETTID   // Set child thread ID in |ptid| of the parent.
    | CLONE_CHILD_CLEARTID; // Let the kernel clear the tid address
                            // wake the joining thread.
// TODO: Add the CLONE_SETTLS flag and setup the TLS area correctly
// when making the clone syscall.

static inline cpp::ErrorOr<void *> alloc_stack(size_t size) {
  long mmap_result =
      __llvm_libc::syscall(MMAP_SYSCALL_NUMBER,
                           0, // No special address
                           size,
                           PROT_READ | PROT_WRITE,      // Read and write stack
                           MAP_ANONYMOUS | MAP_PRIVATE, // Process private
                           -1, // Not backed by any file
                           0   // No offset
      );
  if (mmap_result < 0 && (uintptr_t(mmap_result) >= UINTPTR_MAX - size))
    return cpp::Error{int(-mmap_result)};
  return reinterpret_cast<void *>(mmap_result);
}

static inline void free_stack(void *stack, size_t size) {
  __llvm_libc::syscall(SYS_munmap, stack, size);
}

template <typename ReturnType> using ThreadRunner = ReturnType(void *);

// We align the start args to 16-byte boundary as we adjust the allocated
// stack memory with its size. We want the adjusted address to be at a
// 16-byte boundary to satisfy the x86_64 and aarch64 ABI requirements.
// If different architecture in future requires higher alignment, then we
// can add a platform specific alignment spec.
template <typename ReturnType> struct alignas(STACK_ALIGNMENT) StartArgs {
  Thread<ReturnType> *thread;
  ThreadRunner<ReturnType> *func;
  void *arg;
};

__attribute__((always_inline)) inline uintptr_t get_start_args_addr() {
  // NOTE: For __builtin_frame_address to work reliably across compilers,
  // architectures and various optimization levels, the TU including this file
  // should be compiled with -fno-omit-frame-pointer.
#ifdef LLVM_LIBC_ARCH_X86_64
  return reinterpret_cast<uintptr_t>(__builtin_frame_address(0))
         // The x86_64 call instruction pushes resume address on to the stack.
         // Next, The x86_64 SysV ABI requires that the frame pointer be pushed
         // on to the stack. So, we have to step past two 64-bit values to get
         // to the start args.
         + sizeof(uintptr_t) * 2;
#elif defined(LLVM_LIBC_ARCH_AARCH64)
  // The frame pointer after cloning the new thread in the Thread::run method
  // is set to the stack pointer where start args are stored. So, we fetch
  // from there.
  return reinterpret_cast<uintptr_t>(__builtin_frame_address(1));
#endif
}

template <typename ReturnType> struct Thread {
private:
  ThreadAttributes<ReturnType> *attrib;
  cpp::Atomic<FutexWordType> *clear_tid;

public:
  Thread() = default;

  static void start_thread() __attribute__((noinline));

  // Return 0 on success or an error value on failure.
  int run(ThreadRunner<ReturnType> *f, void *arg, void *stack, size_t size,
          bool detached = false) {
    bool owned_stack = false;
    if (stack == nullptr) {
      if (size == 0)
        size = DEFAULT_STACK_SIZE;
      auto alloc = alloc_stack(size);
      if (!alloc)
        return alloc.error_code();
      else
        stack = alloc.value();
      owned_stack = true;
    }

    // When the new thread is spawned by the kernel, the new thread gets the
    // stack we pass to the clone syscall. However, this stack is empty and does
    // not have any local vars present in this function. Hence, one cannot
    // pass arguments to the thread start function, or use any local vars from
    // here. So, we pack them into the new stack from where the thread can sniff
    // them out.
    //
    // Likewise, the actual thread state information is also stored on the
    // stack memory.
    uintptr_t adjusted_stack = reinterpret_cast<uintptr_t>(stack) + size -
                               sizeof(StartArgs<ReturnType>) -
                               sizeof(ThreadAttributes<ReturnType>) -
                               sizeof(cpp::Atomic<FutexWordType>);
    adjusted_stack &= ~(uintptr_t(STACK_ALIGNMENT) - 1);

    auto *start_args =
        reinterpret_cast<StartArgs<ReturnType> *>(adjusted_stack);
    start_args->thread = this;
    start_args->func = f;
    start_args->arg = arg;

    attrib = reinterpret_cast<ThreadAttributes<ReturnType> *>(
        adjusted_stack + sizeof(StartArgs<ReturnType>));
    attrib->detach_state =
        uint32_t(detached ? DetachState::DETACHED : DetachState::JOINABLE);
    attrib->stack = stack;
    attrib->stack_size = size;
    attrib->owned_stack = owned_stack;

    clear_tid = reinterpret_cast<cpp::Atomic<FutexWordType> *>(
        adjusted_stack + sizeof(StartArgs<ReturnType>) +
        sizeof(ThreadAttributes<ReturnType>));
    clear_tid->val = CLEAR_TID_VALUE;

    // The clone syscall takes arguments in an architecture specific order.
    // Also, we want the result of the syscall to be in a register as the child
    // thread gets a completely different stack after it is created. The stack
    // variables from this function will not be availalbe to the child thread.
#ifdef LLVM_LIBC_ARCH_X86_64
    long register clone_result asm("rax");
    clone_result = __llvm_libc::syscall(
        SYS_clone, CLONE_SYSCALL_FLAGS, adjusted_stack,
        &attrib->tid,    // The address where the child tid is written
        &clear_tid->val, // The futex where the child thread status is signalled
        0                // Set TLS to null for now.
    );
#elif defined(LLVM_LIBC_ARCH_AARCH64)
    long register clone_result asm("x0");
    clone_result = __llvm_libc::syscall(
        SYS_clone, CLONE_SYSCALL_FLAGS, adjusted_stack,
        &attrib->tid,   // The address where the child tid is written
        0,              // Set TLS to null for now.
        &clear_tid->val // The futex where the child thread status is signalled
    );
#else
#error "Unsupported architecture for the clone syscall."
#endif

    if (clone_result == 0) {
#ifdef LLVM_LIBC_ARCH_AARCH64
      // We set the frame pointer to be the same as the "sp" so that start args
      // can be sniffed out from start_thread.
      __arm_wsr64("x29", __arm_rsr64("sp"));
#endif
      start_thread();
    } else if (clone_result < 0) {
      if (attrib->owned_stack)
        free_stack(attrib->stack, attrib->stack_size);
      return -clone_result;
    }

    return 0;
  }

  int join(ReturnType *retval) {
    wait();

    *retval = attrib->retval;
    if (attrib->owned_stack)
      free_stack(attrib->stack, attrib->stack_size);

    return 0;
  }

  // Detach a joinable thread.
  //
  // This method does not have error return value. However, the type of detach
  // is returned to help with testing.
  int detach() {
    uint32_t joinable_state = uint32_t(DetachState::JOINABLE);
    if (attrib->detach_state.compare_exchange_strong(
            joinable_state, uint32_t(DetachState::DETACHED))) {
      return int(DetachType::SIMPLE);
    }

    // If the thread was already detached, then the detach method should not
    // be called at all. If the thread is exiting, then we wait for it to exit
    // and free up resources.
    wait();

    if (attrib->owned_stack)
      free_stack(attrib->stack, attrib->stack_size);
    return int(DetachType::CLEANUP);
  }

  // Wait for the thread to finish. This method can only be called
  // if:
  // 1. A detached thread is guaranteed to be running.
  // 2. A joinable thread has not been detached or joined. As long as it has
  //    not been detached or joined, wait can be called multiple times.
  //
  // Also, only one thread can wait and expect to get woken up when the thread
  // finishes.
  //
  // NOTE: This function is to be used for testing only. There is no standard
  // which requires exposing it via a public API.
  void wait() {
    // The kernel should set the value at the clear tid address to zero.
    // If not, it is a spurious wake and we should continue to wait on
    // the futex.
    while (clear_tid->load() != 0) {
      // We cannot do a FUTEX_WAIT_PRIVATE here as the kernel does a
      // FUTEX_WAKE and not a FUTEX_WAKE_PRIVATE.
      __llvm_libc::syscall(SYS_futex, &clear_tid->val, FUTEX_WAIT,
                           CLEAR_TID_VALUE, nullptr);
    }
  }
};

template <typename ReturnType>
__attribute__((noinline)) void Thread<ReturnType>::start_thread() {
  auto *start_args =
      reinterpret_cast<StartArgs<ReturnType> *>(get_start_args_addr());
  auto *thread = start_args->thread;
  ReturnType retval = thread->attrib->retval =
      start_args->func(start_args->arg);

  uint32_t joinable_state = uint32_t(DetachState::JOINABLE);
  if (!thread->attrib->detach_state.compare_exchange_strong(
          joinable_state, uint32_t(DetachState::EXITING))) {
    // Thread is detached so cleanup the resources.
    if (thread->attrib->owned_stack)
      free_stack(thread->attrib->stack, thread->attrib->stack_size);
  }

  __llvm_libc::syscall(SYS_exit, retval);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_LINUX_THREAD_H
