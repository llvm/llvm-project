//===-- hwasan_thread.h -----------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file is a part of HWAddressSanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef HWASAN_THREAD_H
#define HWASAN_THREAD_H

#include "hwasan_allocator.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_thread_registry.h"

namespace __hwasan {

class Thread;

class ThreadContext : public ThreadContextBase {
 public:
  explicit ThreadContext(int tid)
      : ThreadContextBase(tid), thread(nullptr){}

  Thread *thread;

  void OnCreated(void *arg) override;
  void OnFinished() override;

  struct Args {
    Thread *thread;
  };
};

// We want this to be small.
COMPILER_CHECK(sizeof(ThreadContext) <= 256);

class Thread {
 public:
  static Thread *Create(thread_callback_t start_routine, void *arg);
  void Destroy();

  void Init();
  thread_return_t ThreadStart();

  uptr stack_top() { return stack_top_; }
  uptr stack_bottom() { return stack_bottom_; }
  uptr tls_begin() { return tls_begin_; }
  uptr tls_end() { return tls_end_; }
  bool IsMainThread() { return start_routine_ == nullptr; }

  bool AddrIsInStack(uptr addr) {
    return addr >= stack_bottom_ && addr < stack_top_;
  }

  bool InSignalHandler() { return in_signal_handler_; }
  void EnterSignalHandler() { in_signal_handler_++; }
  void LeaveSignalHandler() { in_signal_handler_--; }

  bool InSymbolizer() { return in_symbolizer_; }
  void EnterSymbolizer() { in_symbolizer_++; }
  void LeaveSymbolizer() { in_symbolizer_--; }

  bool InInterceptorScope() { return in_interceptor_scope_; }
  void EnterInterceptorScope() { in_interceptor_scope_++; }
  void LeaveInterceptorScope() { in_interceptor_scope_--; }

  HwasanThreadLocalMallocStorage &malloc_storage() { return malloc_storage_; }
  HeapAllocationsRingBuffer *heap_allocations() {
    return heap_allocations_;
  }

  void set_context(ThreadContext *context) { context_ = context; }
  const ThreadContext *context() const { return context_; }

  tag_t GenerateRandomTag();

  int destructor_iterations_;
  void DisableTagging() { tagging_disabled_++; }
  void EnableTagging() { tagging_disabled_--; }
  bool TaggingIsDisabled() const { return tagging_disabled_; }

 private:
  // NOTE: There is no Thread constructor. It is allocated
  // via mmap() and *must* be valid in zero-initialized state.
  void SetThreadStackAndTls();
  void ClearShadowForThreadStackAndTLS();
  thread_callback_t start_routine_;
  void *arg_;
  uptr stack_top_;
  uptr stack_bottom_;
  uptr tls_begin_;
  uptr tls_end_;

  unsigned in_signal_handler_;
  unsigned in_symbolizer_;
  unsigned in_interceptor_scope_;

  u32 random_state_;
  u32 random_buffer_;

  HwasanThreadLocalMallocStorage malloc_storage_;
  HeapAllocationsRingBuffer *heap_allocations_;

  u32 tid_;
  ThreadContext *context_;

  u32 tagging_disabled_;  // if non-zero, malloc uses zero tag in this thread.
};

Thread *GetCurrentThread();
void SetCurrentThread(Thread *t);

// Returns the ThreadRegistry singleton.
ThreadRegistry &GetThreadRegistry();

struct ScopedTaggingDisabler {
  ScopedTaggingDisabler() { GetCurrentThread()->DisableTagging(); }
  ~ScopedTaggingDisabler() { GetCurrentThread()->EnableTagging(); }
};

// Returns the ThreadRegistry singleton.

} // namespace __hwasan

#endif // HWASAN_THREAD_H
