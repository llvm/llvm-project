//=-- dsan_thread.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// See dsan_thread.h for details.
//
//===----------------------------------------------------------------------===//

#include "dsan_thread.h"

#include "dsan.h"
#include "dsan_allocator.h"
#include "dsan_common.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_thread_history.h"
#include "sanitizer_common/sanitizer_thread_registry.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

namespace __dsan {

static ThreadRegistry* thread_registry;
static ThreadArgRetval* thread_arg_retval;

static Mutex mu_for_thread_context;
static LowLevelAllocator allocator_for_thread_context;

static ThreadContextBase* CreateThreadContext(u32 tid) {
  Lock lock(&mu_for_thread_context);
  return new (allocator_for_thread_context) ThreadContext(tid);
}

void InitializeThreads() {
  alignas(alignof(ThreadRegistry)) static char
      thread_registry_placeholder[sizeof(ThreadRegistry)];
  thread_registry =
      new (thread_registry_placeholder) ThreadRegistry(CreateThreadContext);

  alignas(alignof(ThreadArgRetval)) static char
      thread_arg_retval_placeholder[sizeof(ThreadArgRetval)];
  thread_arg_retval = new (thread_arg_retval_placeholder) ThreadArgRetval();
}

ThreadArgRetval& GetThreadArgRetval() { return *thread_arg_retval; }

ThreadContextDsanBase::ThreadContextDsanBase(int tid)
    : ThreadContextBase(tid) {}

void ThreadContextDsanBase::OnStarted(void* arg) {
  SetCurrentThread(this);
  AllocatorThreadStart();
}

void ThreadContextDsanBase::OnFinished() {
  AllocatorThreadFinish();
  DTLS_Destroy();
  SetCurrentThread(nullptr);
}

u32 ThreadCreate(u32 parent_tid, bool detached, void* arg) {
  return thread_registry->CreateThread(0, detached, parent_tid, arg);
}

void ThreadContextDsanBase::ThreadStart(u32 tid, ThreadID os_id,
                                        ThreadType thread_type, void* arg) {
  thread_registry->StartThread(tid, os_id, thread_type, arg);
}

void ThreadFinish() { thread_registry->FinishThread(GetCurrentThreadId()); }

void EnsureMainThreadIDIsCorrect() {
  if (GetCurrentThreadId() == kMainTid)
    GetCurrentThread()->os_id = GetTid();
}

///// Interface to the common DSan module. /////

void GetThreadExtraStackRangesLocked(ThreadID os_id,
                                     InternalMmapVector<Range>* ranges) {}
void GetThreadExtraStackRangesLocked(InternalMmapVector<Range>* ranges) {}

void LockThreads() {
  thread_registry->Lock();
  thread_arg_retval->Lock();
}

void UnlockThreads() {
  thread_arg_retval->Unlock();
  thread_registry->Unlock();
}

ThreadRegistry* GetDsanThreadRegistryLocked() {
  thread_registry->CheckLocked();
  return thread_registry;
}

void GetRunningThreadsLocked(InternalMmapVector<ThreadID>* threads) {
  GetDsanThreadRegistryLocked()->RunCallbackForEachThreadLocked(
      [](ThreadContextBase* tctx, void* threads) {
        if (tctx->status == ThreadStatusRunning) {
          reinterpret_cast<InternalMmapVector<ThreadID>*>(threads)->push_back(
              tctx->os_id);
        }
      },
      threads);
}

void PrintThreads() {
  InternalScopedString out;
  PrintThreadHistory(*thread_registry, out);
  Report("%s\n", out.data());
}

void GetAdditionalThreadContextPtrsLocked(InternalMmapVector<uptr>* ptrs) {
  GetThreadArgRetval().GetAllPtrsLocked(ptrs);
}

}  // namespace __dsan
