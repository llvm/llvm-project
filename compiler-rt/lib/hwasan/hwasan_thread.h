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

namespace __hwasan {

struct ThreadStartArg {
  thread_callback_t callback;
  void *param;
};

class Thread {
 public:
  static Thread *Create(thread_callback_t start_routine, void *arg);
  void Destroy();

  void Init();

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

  tag_t GenerateRandomTag();

  int destructor_iterations_;
  void DisableTagging() { tagging_disabled_++; }
  void EnableTagging() { tagging_disabled_--; }
  bool TaggingIsDisabled() const { return tagging_disabled_; }

  template <class CB>
  static void VisitAllLiveThreads(CB cb) {
    SpinMutexLock l(&thread_list_mutex);
    Thread *t = main_thread;
    while (t) {
      cb(t);
      t = t->next_;
    }
  }

  // Return a scratch ThreadStartArg object to be used in
  // pthread_create interceptor.
  ThreadStartArg *thread_start_arg() { return &thread_start_arg_; }

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

  static void InsertIntoThreadList(Thread *t);
  static void RemoveFromThreadList(Thread *t);
  Thread *next_;  // All live threads form a linked list.
  static SpinMutex thread_list_mutex;
  static Thread *main_thread;

  u32 tagging_disabled_;  // if non-zero, malloc uses zero tag in this thread.

  ThreadStartArg thread_start_arg_;
};

Thread *GetCurrentThread();
void SetCurrentThread(Thread *t);

struct ScopedTaggingDisabler {
  ScopedTaggingDisabler() { GetCurrentThread()->DisableTagging(); }
  ~ScopedTaggingDisabler() { GetCurrentThread()->EnableTagging(); }
};

} // namespace __hwasan

#endif // HWASAN_THREAD_H
