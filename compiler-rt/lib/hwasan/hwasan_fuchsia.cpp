//===-- hwasan_fuchsia.cpp --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file is a part of HWAddressSanitizer and contains Fuchsia-specific
/// code.
///
//===----------------------------------------------------------------------===//

#include "sanitizer_common/sanitizer_fuchsia.h"
#if SANITIZER_FUCHSIA

#include "hwasan.h"
#include "hwasan_interface_internal.h"
#include "hwasan_report.h"
#include "hwasan_thread.h"
#include "hwasan_thread_list.h"

// This TLS variable contains the location of the stack ring buffer and can be
// used to always find the hwasan thread object associated with the current
// running thread.
[[gnu::tls_model("initial-exec")]]
SANITIZER_INTERFACE_ATTRIBUTE
THREADLOCAL uptr __hwasan_tls;

namespace __hwasan {

// These are known parameters passed to the hwasan runtime on thread creation.
struct Thread::InitState {
  uptr stack_bottom, stack_top;
};

static void FinishThreadInitialization(Thread *thread);

void InitThreads() {
  // This is the minimal alignment needed for the storage where hwasan threads
  // and their stack ring buffers are placed. This alignment is necessary so the
  // stack ring buffer can perform a simple calculation to get the next element
  // in the RB. The instructions for this calculation are emitted by the
  // compiler. (Full explanation in hwasan_thread_list.h.)
  uptr alloc_size = UINT64_C(1) << kShadowBaseAlignment;
  uptr thread_start = reinterpret_cast<uptr>(
      MmapAlignedOrDieOnFatalError(alloc_size, alloc_size, __func__));

  InitThreadList(thread_start, alloc_size);

  // Create the hwasan thread object for the current (main) thread. Stack info
  // for this thread is known from information passed via
  // __sanitizer_startup_hook.
  const Thread::InitState state = {
      .stack_bottom = __sanitizer::MainThreadStackBase,
      .stack_top =
          __sanitizer::MainThreadStackBase + __sanitizer::MainThreadStackSize,
  };
  FinishThreadInitialization(hwasanThreadList().CreateCurrentThread(&state));
}

uptr *GetCurrentThreadLongPtr() { return &__hwasan_tls; }

// This is called from the parent thread before the new thread is created. Here
// we can propagate known info like the stack bounds to Thread::Init before
// jumping into the thread. We cannot initialize the stack ring buffer yet since
// we have not entered the new thread.
static void *BeforeThreadCreateHook(uptr user_id, bool detached,
                                    const char *name, uptr stack_bottom,
                                    uptr stack_size) {
  const Thread::InitState state = {
      .stack_bottom = stack_bottom,
      .stack_top = stack_bottom + stack_size,
  };
  return hwasanThreadList().CreateCurrentThread(&state);
}

// This sets the stack top and bottom according to the InitState passed to
// CreateCurrentThread above.
void Thread::InitStackAndTls(const InitState *state) {
  CHECK_NE(state->stack_bottom, 0);
  CHECK_NE(state->stack_top, 0);
  stack_bottom_ = state->stack_bottom;
  stack_top_ = state->stack_top;
  tls_end_ = tls_begin_ = 0;
}

// This is called after creating a new thread with the pointer returned by
// BeforeThreadCreateHook. We are still in the creating thread and should check
// if it was actually created correctly.
static void ThreadCreateHook(void *hook, bool aborted) {
  Thread *thread = static_cast<Thread *>(hook);
  if (!aborted) {
    // The thread was created successfully.
    // ThreadStartHook can already be running in the new thread.
  } else {
    // The thread wasn't created after all.
    // Clean up everything we set up in BeforeThreadCreateHook.
    atomic_signal_fence(memory_order_seq_cst);
    hwasanThreadList().ReleaseThread(thread);
  }
}

// This is called in the newly-created thread before it runs anything else,
// with the pointer returned by BeforeThreadCreateHook (above). Here we can
// setup the stack ring buffer.
static void ThreadStartHook(void *hook, thrd_t self) {
  Thread *thread = static_cast<Thread *>(hook);
  FinishThreadInitialization(thread);
  thread->InitRandomState();
}

// This is the function that sets up the stack ring buffer and enables us to use
// GetCurrentThread. This function should only be called while IN the thread
// that we want to create the hwasan thread object for so __hwasan_tls can be
// properly referenced.
static void FinishThreadInitialization(Thread *thread) {
  CHECK_NE(thread, nullptr);

  // The ring buffer is located immediately before the thread object.
  uptr stack_buffer_size = hwasanThreadList().GetRingBufferSize();
  uptr stack_buffer_start = reinterpret_cast<uptr>(thread) - stack_buffer_size;
  thread->InitStackRingBuffer(stack_buffer_start, stack_buffer_size);
}

static void ThreadExitHook(void *hook, thrd_t self) {
  Thread *thread = static_cast<Thread *>(hook);
  atomic_signal_fence(memory_order_seq_cst);
  hwasanThreadList().ReleaseThread(thread);
}

}  // namespace __hwasan

extern "C" {

void *__sanitizer_before_thread_create_hook(thrd_t thread, bool detached,
                                            const char *name, void *stack_base,
                                            size_t stack_size) {
  return __hwasan::BeforeThreadCreateHook(
      reinterpret_cast<uptr>(thread), detached, name,
      reinterpret_cast<uptr>(stack_base), stack_size);
}

void __sanitizer_thread_create_hook(void *hook, thrd_t thread, int error) {
  __hwasan::ThreadCreateHook(hook, error != thrd_success);
}

void __sanitizer_thread_start_hook(void *hook, thrd_t self) {
  __hwasan::ThreadStartHook(hook, reinterpret_cast<uptr>(self));
}

void __sanitizer_thread_exit_hook(void *hook, thrd_t self) {
  __hwasan::ThreadExitHook(hook, self);
}

}  // extern "C"

#endif  // SANITIZER_FUCHSIA
