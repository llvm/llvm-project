//=-- lsan_thread.h -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// Thread registry for standalone LSan.
//
//===----------------------------------------------------------------------===//

#ifndef LSAN_THREAD_H
#define LSAN_THREAD_H

#include "sanitizer_common/sanitizer_thread_registry.h"

namespace __lsan {

class ThreadContextLsanBase : public ThreadContextBase {
 public:
  explicit ThreadContextLsanBase(int tid);
  void OnFinished() override;
  uptr stack_begin() { return stack_begin_; }
  uptr stack_end() { return stack_end_; }
  uptr cache_begin() { return cache_begin_; }
  uptr cache_end() { return cache_end_; }

  // The argument is passed on to the subclass's OnStarted member function.
  static void ThreadStart(u32 tid, tid_t os_id, ThreadType thread_type,
                          void *onstarted_arg);

 protected:
  ~ThreadContextLsanBase() {}
  uptr stack_begin_ = 0;
  uptr stack_end_ = 0;
  uptr cache_begin_ = 0;
  uptr cache_end_ = 0;
};

// This subclass of ThreadContextLsanBase is declared in an OS-specific header.
class ThreadContext;

void InitializeThreadRegistry();
void InitializeMainThread();

u32 ThreadCreate(u32 tid, bool detached, void *arg = nullptr);
void ThreadFinish();

u32 GetCurrentThread();
void SetCurrentThread(u32 tid);
ThreadContext *CurrentThreadContext();
void EnsureMainThreadIDIsCorrect();

}  // namespace __lsan

#endif  // LSAN_THREAD_H
