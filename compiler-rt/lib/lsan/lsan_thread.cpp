//=-- lsan_thread.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of LeakSanitizer.
// See lsan_thread.h for details.
//
//===----------------------------------------------------------------------===//

#include "lsan_thread.h"

#include "lsan.h"
#include "lsan_allocator.h"
#include "lsan_common.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_placement_new.h"
#include "sanitizer_common/sanitizer_thread_registry.h"
#include "sanitizer_common/sanitizer_tls_get_addr.h"

namespace __lsan {

static ThreadRegistry *thread_registry;

static ThreadContextBase *CreateThreadContext(u32 tid) {
  void *mem = MmapOrDie(sizeof(ThreadContext), "ThreadContext");
  return new (mem) ThreadContext(tid);
}

void InitializeThreadRegistry() {
  static ALIGNED(64) char thread_registry_placeholder[sizeof(ThreadRegistry)];
  thread_registry =
      new (thread_registry_placeholder) ThreadRegistry(CreateThreadContext);
}

ThreadContextLsanBase::ThreadContextLsanBase(int tid)
    : ThreadContextBase(tid) {}

void ThreadContextLsanBase::OnStarted(void *arg) { SetCurrentThread(tid); }

void ThreadContextLsanBase::OnFinished() {
  AllocatorThreadFinish();
  DTLS_Destroy();
  SetCurrentThread(kInvalidTid);
}

u32 ThreadCreate(u32 parent_tid, bool detached, void *arg) {
  return thread_registry->CreateThread(0, detached, parent_tid, arg);
}

void ThreadContextLsanBase::ThreadStart(u32 tid, tid_t os_id,
                                        ThreadType thread_type, void *arg) {
  thread_registry->StartThread(tid, os_id, thread_type, arg);
}

void ThreadFinish() { thread_registry->FinishThread(GetCurrentThread()); }

ThreadContext *CurrentThreadContext() {
  if (!thread_registry)
    return nullptr;
  if (GetCurrentThread() == kInvalidTid)
    return nullptr;
  // No lock needed when getting current thread.
  return (ThreadContext *)thread_registry->GetThreadLocked(GetCurrentThread());
}

void EnsureMainThreadIDIsCorrect() {
  if (GetCurrentThread() == kMainTid)
    CurrentThreadContext()->os_id = GetTid();
}

///// Interface to the common LSan module. /////

void GetThreadExtraStackRangesLocked(tid_t os_id,
                                     InternalMmapVector<Range> *ranges) {}
void GetThreadExtraStackRangesLocked(InternalMmapVector<Range> *ranges) {}

void LockThreadRegistry() { thread_registry->Lock(); }

void UnlockThreadRegistry() { thread_registry->Unlock(); }

ThreadRegistry *GetLsanThreadRegistryLocked() {
  thread_registry->CheckLocked();
  return thread_registry;
}

void GetRunningThreadsLocked(InternalMmapVector<tid_t> *threads) {
  GetLsanThreadRegistryLocked()->RunCallbackForEachThreadLocked(
      [](ThreadContextBase *tctx, void *threads) {
        if (tctx->status == ThreadStatusRunning) {
          reinterpret_cast<InternalMmapVector<tid_t> *>(threads)->push_back(
              tctx->os_id);
        }
      },
      threads);
}

}  // namespace __lsan
