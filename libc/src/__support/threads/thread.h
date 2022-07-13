//===--- A platform independent indirection for a thread class --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_H
#define LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_H

#include "src/__support/CPP/atomic.h"
#include "src/__support/architectures.h"

#include <stddef.h> // For size_t
#include <stdint.h>

namespace __llvm_libc {

using ThreadRunnerPosix = void *(void *);
using ThreadRunnerStdc = int(void *);

union ThreadRunner {
  ThreadRunnerPosix *posix_runner;
  ThreadRunnerStdc *stdc_runner;
};

union ThreadReturnValue {
  void *posix_retval;
  int stdc_retval;
  constexpr ThreadReturnValue() : posix_retval(nullptr) {}
};

#if (defined(LLVM_LIBC_ARCH_AARCH64) || defined(LLVM_LIBC_ARCH_X86_64))
constexpr unsigned int STACK_ALIGNMENT = 16;
#endif
// TODO: Provide stack alignment requirements for other architectures.

enum class DetachState : uint32_t {
  JOINABLE = 0x11,
  EXITING = 0x22,
  DETACHED = 0x33
};

enum class ThreadStyle : uint8_t { POSIX = 0x1, STDC = 0x2 };

// Detach type is useful in testing the detach operation.
enum class DetachType : int {
  // Indicates that the detach operation just set the detach state to DETACHED
  // and returned.
  SIMPLE = 1,

  // Indicates that the detach operation performed thread cleanup.
  CLEANUP = 2
};

// A data type to hold common thread attributes which have to be stored as
// thread state. Note that this is different from public attribute types like
// pthread_attr_t which might contain information which need not be saved as
// part of a thread's state. For example, the stack guard size.
//
// Thread attributes are typically stored on the stack. So, we align as required
// for the target architecture.
struct alignas(STACK_ALIGNMENT) ThreadAttributes {
  // We want the "detach_state" attribute to be an atomic value as it could be
  // updated by one thread while the self thread is reading it. It is a tristate
  // variable with the following state transitions:
  // 1. The a thread is created in a detached state, then user code should never
  //    call a detach or join function. Calling either of them can lead to
  //    undefined behavior.
  //    The value of |detach_state| is expected to be DetachState::DETACHED for
  //    its lifetime.
  // 2. If a thread is created in a joinable state, |detach_state| will start
  //    with the value DetachState::JOINABLE. Another thread can detach this
  //    thread before it exits. The state transitions will as follows:
  //      (a) If the detach method sees the state as JOINABLE, then it will
  //          compare exchange to a state of DETACHED. The thread will clean
  //          itself up after it finishes.
  //      (b) If the detach method does not see JOINABLE in (a), then it will
  //          conclude that the thread is EXITING and will wait until the thread
  //          exits. It will clean up the thread resources once the thread
  //          exits.
  cpp::Atomic<uint32_t> detach_state;
  void *stack; // Pointer to the thread stack
  void *tls;
  unsigned long long stack_size; // Size of the stack
  unsigned char owned_stack; // Indicates if the thread owns this stack memory
  int tid;
  ThreadStyle style;
  ThreadReturnValue retval;
  void *platform_data;

  constexpr ThreadAttributes()
      : detach_state(uint32_t(DetachState::DETACHED)), stack(nullptr),
        tls(nullptr), stack_size(0), owned_stack(false), tid(-1),
        style(ThreadStyle::POSIX), retval(),
        platform_data(nullptr) {
  }
};

struct Thread {
  ThreadAttributes *attrib;

  constexpr Thread() : attrib(nullptr) {}
  constexpr Thread(ThreadAttributes *attr) : attrib(attr) {}

  int run(ThreadRunnerPosix *func, void *arg, void *stack, size_t size,
          bool detached = false) {
    ThreadRunner runner;
    runner.posix_runner = func;
    return run(ThreadStyle::POSIX, runner, arg, stack, size, detached);
  }

  int run(ThreadRunnerStdc *func, void *arg, void *stack, size_t size,
          bool detached = false) {
    ThreadRunner runner;
    runner.stdc_runner = func;
    return run(ThreadStyle::STDC, runner, arg, stack, size, detached);
  }

  int join(int *val) {
    ThreadReturnValue retval;
    int status = join(retval);
    if (status != 0)
      return status;
    *val = retval.stdc_retval;
    return 0;
  }

  int join(void **val) {
    ThreadReturnValue retval;
    int status = join(retval);
    if (status != 0)
      return status;
    *val = retval.posix_retval;
    return 0;
  }

  // Platform should implement the functions below.

  // Return 0 on success or an error value on failure.
  int run(ThreadStyle style, ThreadRunner runner, void *arg, void *stack,
          size_t stack_size, bool detached);

  // Return 0 on success or an error value on failure.
  int join(ThreadReturnValue &retval);

  // Detach a joinable thread.
  //
  // This method does not have error return value. However, the type of detach
  // is returned to help with testing.
  int detach();

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
  void wait();
};

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_SUPPORT_THREADS_THREAD_H
