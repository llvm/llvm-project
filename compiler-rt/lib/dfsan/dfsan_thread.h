//===-- dfsan_thread.h -------------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DataFlowSanitizer.
//
//===----------------------------------------------------------------------===//

#ifndef DFSAN_THREAD_H
#define DFSAN_THREAD_H

#include "sanitizer_common/sanitizer_common.h"

namespace __dfsan {

class DFsanThread {
 public:
  // NOTE: There is no DFsanThread constructor. It is allocated
  // via mmap() and *must* be valid in zero-initialized state.

  static DFsanThread *Create(void *start_routine_trampoline,
                             thread_callback_t start_routine, void *arg);
  static void TSDDtor(void *tsd);
  void Destroy();

  void Init();  // Should be called from the thread itself.
  thread_return_t ThreadStart();

  uptr stack_top();
  uptr stack_bottom();
  bool IsMainThread() { return start_routine_ == nullptr; }

  bool InSignalHandler() { return in_signal_handler_; }
  void EnterSignalHandler() { in_signal_handler_++; }
  void LeaveSignalHandler() { in_signal_handler_--; }

  int destructor_iterations_;

 private:
  void SetThreadStackAndTls();
  struct StackBounds {
    uptr bottom;
    uptr top;
  };
  StackBounds GetStackBounds() const;

  bool AddrIsInStack(uptr addr);

  void *start_routine_trampoline_;
  thread_callback_t start_routine_;
  void *arg_;

  StackBounds stack_;

  unsigned in_signal_handler_;
};

DFsanThread *GetCurrentThread();
void SetCurrentThread(DFsanThread *t);
void DFsanTSDInit(void (*destructor)(void *tsd));
void DFsanTSDDtor(void *tsd);

}  // namespace __dfsan

#endif  // DFSAN_THREAD_H
