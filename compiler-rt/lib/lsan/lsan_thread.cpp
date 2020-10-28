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

static const uptr kMaxThreads = 1 << 13;
static const uptr kThreadQuarantineSize = 64;

void InitializeThreadRegistry() {
  static ALIGNED(64) char thread_registry_placeholder[sizeof(ThreadRegistry)];
  thread_registry = new (thread_registry_placeholder)
      ThreadRegistry(CreateThreadContext, kMaxThreads, kThreadQuarantineSize);
}

ThreadContextLsanBase::ThreadContextLsanBase(int tid)
    : ThreadContextBase(tid) {}

void ThreadContextLsanBase::OnFinished() {
  AllocatorThreadFinish();
  DTLS_Destroy();
}

u32 ThreadCreate(u32 parent_tid, uptr user_id, bool detached, void *arg) {
  return thread_registry->CreateThread(user_id, detached, parent_tid, arg);
}

void ThreadContextLsanBase::ThreadStart(u32 tid, tid_t os_id,
                                        ThreadType thread_type, void *arg) {
  thread_registry->StartThread(tid, os_id, thread_type, arg);
  SetCurrentThread(tid);
}

void ThreadFinish() {
  thread_registry->FinishThread(GetCurrentThread());
  SetCurrentThread(kInvalidTid);
}

ThreadContext *CurrentThreadContext() {
  if (!thread_registry)
    return nullptr;
  if (GetCurrentThread() == kInvalidTid)
    return nullptr;
  // No lock needed when getting current thread.
  return (ThreadContext *)thread_registry->GetThreadLocked(GetCurrentThread());
}

static bool FindThreadByUid(ThreadContextBase *tctx, void *arg) {
  uptr uid = (uptr)arg;
  if (tctx->user_id == uid && tctx->status != ThreadStatusInvalid) {
    return true;
  }
  return false;
}

u32 ThreadTid(uptr uid) {
  return thread_registry->FindThread(FindThreadByUid, (void *)uid);
}

void ThreadDetach(u32 tid) {
  CHECK_NE(tid, kInvalidTid);
  thread_registry->DetachThread(tid, /* arg */ nullptr);
}

void ThreadJoin(u32 tid) {
  CHECK_NE(tid, kInvalidTid);
  thread_registry->JoinThread(tid, /* arg */ nullptr);
}

void EnsureMainThreadIDIsCorrect() {
  if (GetCurrentThread() == 0)
    CurrentThreadContext()->os_id = GetTid();
}

///// Interface to the common LSan module. /////

void ForEachExtraStackRange(tid_t os_id, RangeIteratorCallback callback,
                            void *arg) {}

void LockThreadRegistry() { thread_registry->Lock(); }

void UnlockThreadRegistry() { thread_registry->Unlock(); }

ThreadRegistry *GetThreadRegistryLocked() {
  thread_registry->CheckLocked();
  return thread_registry;
}

}  // namespace __lsan
