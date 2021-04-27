//===-- sanitizer_thread_registry.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between sanitizer tools.
//
// General thread bookkeeping functionality.
//===----------------------------------------------------------------------===//

#include "sanitizer_thread_registry.h"

namespace __sanitizer {

ThreadContextBase::ThreadContextBase(Tid tid)
    : tid(tid),
      os_id(0),
      user_id(0),
      status(ThreadStatusInvalid),
      detached(false),
      thread_type(ThreadType::Regular),
      parent_tid(kInvalidTid) {
  name[0] = '\0';
  atomic_store(&thread_destroyed, 0, memory_order_release);
}

ThreadContextBase::~ThreadContextBase() {
  // ThreadContextBase should never be deleted.
  CHECK(0);
}

void ThreadContextBase::SetName(const char *new_name) {
  name[0] = '\0';
  if (new_name) {
    internal_strncpy(name, new_name, sizeof(name));
    name[sizeof(name) - 1] = '\0';
  }
}

void ThreadContextBase::SetDead() {
  CHECK(status == ThreadStatusRunning ||
        status == ThreadStatusFinished);
  status = ThreadStatusDead;
  user_id = 0;
  OnDead();
}

void ThreadContextBase::SetDestroyed() {
  atomic_store(&thread_destroyed, 1, memory_order_release);
}

bool ThreadContextBase::GetDestroyed() {
  return !!atomic_load(&thread_destroyed, memory_order_acquire);
}

void ThreadContextBase::SetJoined(void *arg) {
  // FIXME(dvyukov): print message and continue (it's user error).
  CHECK_EQ(false, detached);
  CHECK_EQ(ThreadStatusFinished, status);
  status = ThreadStatusDead;
  user_id = 0;
  OnJoined(arg);
}

void ThreadContextBase::SetFinished() {
  // ThreadRegistry::FinishThread calls here in ThreadStatusCreated state
  // for a thread that never actually started.  In that case the thread
  // should go to ThreadStatusFinished regardless of whether it was created
  // as detached.
  if (!detached || status == ThreadStatusCreated) status = ThreadStatusFinished;
  OnFinished();
}

void ThreadContextBase::SetStarted(tid_t _os_id, ThreadType _thread_type,
                                   void *arg) {
  status = ThreadStatusRunning;
  os_id = _os_id;
  thread_type = _thread_type;
  OnStarted(arg);
}

void ThreadContextBase::SetCreated(uptr _user_id, bool _detached,
                                   Tid _parent_tid, void *arg) {
  status = ThreadStatusCreated;
  user_id = _user_id;
  detached = _detached;
  // Parent tid makes no sense for the main thread.
  if (tid != kMainTid)
    parent_tid = _parent_tid;
  OnCreated(arg);
}

void ThreadContextBase::Reset() {
  status = ThreadStatusInvalid;
  SetName(0);
  atomic_store(&thread_destroyed, 0, memory_order_release);
  OnReset();
}

// ThreadRegistry implementation.

ThreadRegistry::ThreadRegistry(ThreadContextFactory factory)
    : context_factory_(factory), mtx_(MutexThreadRegistry), total_threads_(), running_threads_() {
  threads_.Initialize(1024);
}

Tid ThreadRegistry::CreateThread(uptr user_id, bool detached, Tid parent_tid,
                                 void *arg) {
  BlockingMutexLock l(&mtx_);
  Tid tid = static_cast<Tid>(threads_.size());
  ThreadContextBase *tctx = context_factory_(tid);
  threads_.push_back(tctx);
  CHECK_EQ(tctx->status, ThreadStatusInvalid);
  atomic_store_relaxed(&total_threads_, atomic_load_relaxed(&total_threads_) + 1);
  tctx->SetCreated(user_id, detached, parent_tid, arg);
  return tid;
}

void ThreadRegistry::RunCallbackForEachThreadLocked(ThreadCallback cb,
                                                    void *arg) {
  CheckLocked();
  for (auto tctx : threads_) {
    if (tctx)
      cb(tctx, arg);
  }
}

Tid ThreadRegistry::FindThread(FindThreadCallback cb, void *arg) {
  BlockingMutexLock l(&mtx_);
  for (auto tctx : threads_) {
    if (tctx && cb(tctx, arg))
      return tctx->tid;
  }
  return kInvalidTid;
}

ThreadContextBase *
ThreadRegistry::FindThreadContextLocked(FindThreadCallback cb, void *arg) {
  CheckLocked();
  for (auto tctx : threads_) {
    if (tctx && cb(tctx, arg))
      return tctx;
  }
  return 0;
}

static bool FindThreadContextByOsIdCallback(ThreadContextBase *tctx,
                                            void *arg) {
  return (tctx->os_id == (uptr)arg && tctx->status != ThreadStatusInvalid &&
      tctx->status != ThreadStatusDead);
}

ThreadContextBase *ThreadRegistry::FindThreadContextByOsIDLocked(tid_t os_id) {
  return FindThreadContextLocked(FindThreadContextByOsIdCallback,
                                 (void *)os_id);
}

void ThreadRegistry::SetThreadName(Tid tid, const char *name) {
  BlockingMutexLock l(&mtx_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_EQ(SANITIZER_FUCHSIA ? ThreadStatusCreated : ThreadStatusRunning,
           tctx->status);
  tctx->SetName(name);
}

void ThreadRegistry::SetThreadNameByUserId(uptr user_id, const char *name) {
  BlockingMutexLock l(&mtx_);
  for (auto tctx : threads_) {
    if (tctx != 0 && tctx->user_id == user_id &&
        tctx->status != ThreadStatusInvalid) {
      tctx->SetName(name);
      return;
    }
  }
}

void ThreadRegistry::DetachThread(Tid tid, void *arg) {
  BlockingMutexLock l(&mtx_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  if (tctx->status == ThreadStatusInvalid) {
    Report("%s: Detach of non-existent thread\n", SanitizerToolName);
    return;
  }
  tctx->OnDetached(arg);
  if (tctx->status == ThreadStatusFinished) {
    tctx->SetDead();
  } else {
    tctx->detached = true;
  }
}

void ThreadRegistry::JoinThread(Tid tid, void *arg) {
  for (;; internal_sched_yield()) {
    BlockingMutexLock l(&mtx_);
    ThreadContextBase *tctx = threads_[tid];
    CHECK_NE(tctx, 0);
    if (tctx->status == ThreadStatusInvalid) {
      Report("%s: Join of non-existent thread\n", SanitizerToolName);
      return;
    }
    if (tctx->GetDestroyed()) {
      tctx->SetJoined(arg);
      return;
    }
  }
}

// Normally this is called when the thread is about to exit.  If
// called in ThreadStatusCreated state, then this thread was never
// really started.  We just did CreateThread for a prospective new
// thread before trying to create it, and then failed to actually
// create it, and so never called StartThread.
ThreadStatus ThreadRegistry::FinishThread(Tid tid) {
  BlockingMutexLock l(&mtx_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  bool dead = tctx->detached;
  ThreadStatus prev_status = tctx->status;
  if (tctx->status == ThreadStatusRunning) {
    u32 nthreads = atomic_load_relaxed(&running_threads_);
    CHECK_GT(nthreads, 0);
    atomic_store_relaxed(&running_threads_, nthreads - 1);
  } else {
    // The thread never really existed.
    CHECK_EQ(tctx->status, ThreadStatusCreated);
    dead = true;
  }
  tctx->SetFinished();
  if (dead)
    tctx->SetDead();
  tctx->SetDestroyed();
  return prev_status;
}

void ThreadRegistry::StartThread(Tid tid, tid_t os_id, ThreadType thread_type,
                                 void *arg) {
  BlockingMutexLock l(&mtx_);
  atomic_store_relaxed(&running_threads_, atomic_load_relaxed(&running_threads_) + 1);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_EQ(ThreadStatusCreated, tctx->status);
  tctx->SetStarted(os_id, thread_type, arg);
}

void ThreadRegistry::SetThreadUserId(Tid tid, uptr user_id) {
  BlockingMutexLock l(&mtx_);
  ThreadContextBase *tctx = threads_[tid];
  CHECK_NE(tctx, 0);
  CHECK_NE(tctx->status, ThreadStatusInvalid);
  CHECK_NE(tctx->status, ThreadStatusDead);
  CHECK_EQ(tctx->user_id, 0);
  tctx->user_id = user_id;
}

}  // namespace __sanitizer
