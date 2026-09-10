//=-- dsan_thread.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Thread registry for standalone DSan.
//
//===----------------------------------------------------------------------===//

#ifndef DSAN_THREAD_H
#define DSAN_THREAD_H

#include "sanitizer_common/sanitizer_thread_arg_retval.h"
#include "sanitizer_common/sanitizer_thread_registry.h"

namespace __dsan {

class ThreadContextDsanBase : public ThreadContextBase {
 public:
  explicit ThreadContextDsanBase(int tid);
  void OnStarted(void* arg) override;
  void OnFinished() override;
  uptr stack_begin() { return stack_begin_; }
  uptr stack_end() { return stack_end_; }
  uptr cache_begin() { return cache_begin_; }
  uptr cache_end() { return cache_end_; }

  // The argument is passed on to the subclass's OnStarted member function.
  static void ThreadStart(u32 tid, ThreadID os_id, ThreadType thread_type,
                          void* onstarted_arg);

 protected:
  ~ThreadContextDsanBase() {}
  uptr stack_begin_ = 0;
  uptr stack_end_ = 0;
  uptr cache_begin_ = 0;
  uptr cache_end_ = 0;
};

// This subclass of ThreadContextDsanBase is declared in an OS-specific header.
class ThreadContext;

void InitializeThreads();
void InitializeMainThread();

ThreadRegistry* GetDsanThreadRegistryLocked();
ThreadArgRetval& GetThreadArgRetval();

u32 ThreadCreate(u32 tid, bool detached, void* arg = nullptr);
void ThreadFinish();

ThreadContextDsanBase* GetCurrentThread();
inline u32 GetCurrentThreadId() {
  ThreadContextDsanBase* ctx = GetCurrentThread();
  return ctx ? ctx->tid : kInvalidTid;
}
void SetCurrentThread(ThreadContextDsanBase* tctx);
void EnsureMainThreadIDIsCorrect();

}  // namespace __dsan

#endif  // DSAN_THREAD_H
