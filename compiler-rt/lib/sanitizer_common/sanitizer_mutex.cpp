//===-- sanitizer_mutex.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between AddressSanitizer and ThreadSanitizer
// run-time libraries.
//===----------------------------------------------------------------------===//

#include "sanitizer_mutex.h"
#include "sanitizer_common.h"

namespace __sanitizer {

void StaticSpinMutex::LockSlow() {
  for (int i = 0;; i++) {
    if (i < 1000)
      ;
    else if (i < 1010)
      internal_sched_yield();
    else
      internal_usleep(10);
    if (atomic_load(&state_, memory_order_relaxed) == 0
        && atomic_exchange(&state_, 1, memory_order_acquire) == 0)
        return;
  }
}

#if SANITIZER_DEBUG && !SANITIZER_GO
SANITIZER_WEAK_ATTRIBUTE MutexMeta mutex_meta[] = {{}};
SANITIZER_WEAK_ATTRIBUTE void PrintMutexPC(uptr pc) {}
const int kMutexTypeMax = 20;
StaticSpinMutex mutex_meta_mtx;
int mutex_count = -1;
// The table fixes what mutexes can be locked under what mutexes.
// If the row for MutexFoo contains MutexBar,
// then Bar can be locked while under Foo mutex.
bool mutex_can_lock[kMutexTypeMax][kMutexTypeMax];
bool mutex_can_lock_adj[kMutexTypeMax][kMutexTypeMax];
bool mutex_multi[kMutexTypeMax];

void DebugMutexInit() {
  bool leaf[kMutexTypeMax] = {};
  int cnt[kMutexTypeMax] = {};
  for (int t = 0;; t++) {
    mutex_count = t;
    CHECK_LE(mutex_count, kMutexTypeMax);
    if (!mutex_meta[t].name)
      break;
    CHECK_EQ(t, mutex_meta[t].type);
    for (int j = 0; j < (int)ARRAY_SIZE(mutex_meta[t].can_lock); j++) {
      MutexType z = mutex_meta[t].can_lock[j];
      if (z == MutexInvalid)
        continue;
      if (z == MutexLeaf) {
        CHECK(!leaf[t]);
        leaf[t] = true;
        continue;
      }
      if (z == MutexMulti) {
        mutex_multi[t] = true;
        continue;
      }
      CHECK_LT(z, kMutexTypeMax);
      CHECK(!mutex_can_lock[t][z]);
      mutex_can_lock[t][z] = true;
      cnt[t]++;
    }
  }
  // Add leaf mutexes.
  for (int t = 0; t < mutex_count; t++) {
    if (!leaf[t])
      continue;
    CHECK_EQ(cnt[t], 0);
    for (int z = 0; z < mutex_count; z++) {
      if (z == MutexInvalid || t == z || leaf[z])
        continue;
      CHECK(!mutex_can_lock[z][t]);
      mutex_can_lock[z][t] = true;
    }
  }
  // Build the transitive closure.
  for (int i = 0; i < mutex_count; i++) {
    for (int j = 0; j < mutex_count; j++)
      mutex_can_lock_adj[i][j] = mutex_can_lock[i][j];
  }
  for (int k = 0; k < mutex_count; k++) {
    for (int i = 0; i < mutex_count; i++) {
      for (int j = 0; j < mutex_count; j++) {
        if (mutex_can_lock_adj[i][k] && mutex_can_lock_adj[k][j])
          mutex_can_lock_adj[i][j] = true;
      }
    }
  }
  // Verify that the graph is acyclic.
  for (int i = 0; i < mutex_count; i++) {
    if (mutex_can_lock_adj[i][i]) {
      Printf("Mutex %s participates in a cycle\n", mutex_meta[i].name);
      Die();
    }
  }
}

struct InternalDeadlockDetector {
  struct LockDesc {
    u64 seq;
    uptr pc;
    int recursion;
  };
  int initialized;
  u64 sequence;
  LockDesc locked[kMutexTypeMax];

  void Lock(MutexType type, uptr pc) {
    if (!Initialize(type))
      return;
    CHECK_GT(type, MutexInvalid);
    CHECK_LT(type, mutex_count);
    u64 max_seq = 0;
    MutexType max_idx = MutexInvalid;
    for (int i = 0; i != mutex_count; i++) {
      if (locked[i].seq == 0)
        continue;
      CHECK_NE(locked[i].seq, max_seq);
      if (max_seq < locked[i].seq) {
        max_seq = locked[i].seq;
        max_idx = (MutexType)i;
      }
    }
    if (max_idx == type && mutex_multi[type]) {
      CHECK_EQ(locked[type].seq, max_seq);
      CHECK(locked[type].pc);
      locked[type].recursion++;
      return;
    }
    if (max_idx != MutexInvalid && !mutex_can_lock[max_idx][type]) {
      Printf("%s: internal deadlock:can't lock %s under %s mutex\n",
          SanitizerToolName,
          mutex_meta[type].name,
          mutex_meta[max_idx].name);
      PrintMutexPC(locked[max_idx].pc);
      CHECK(0);
    }
    locked[type].seq = ++sequence;
    locked[type].pc = pc;    
    locked[type].recursion = 1;
  }

  void Unlock(MutexType type) {
    if (!Initialize(type))
      return;
    CHECK(locked[type].seq);
    CHECK_GT(locked[type].recursion, 0);
    if (--locked[type].recursion)
      return;
    locked[type].seq = 0;
    locked[type].pc = 0;
  }
  
  bool Initialize(MutexType type) {
    if (type == MutexUnchecked || type == MutexInvalid)
      return false;
    CHECK_GT(type, MutexInvalid);
    CHECK_LT(type, mutex_count);
    if (initialized != 0)
      return initialized > 0;
    initialized = -1;
    SpinMutexLock lock(&mutex_meta_mtx);
    if (mutex_count < 0)
      DebugMutexInit();
    initialized = mutex_count ? 1 : -1;
    return initialized > 0;
  }
};

THREADLOCAL InternalDeadlockDetector deadlock_detector;

void DebugMutexLock(MutexType type, uptr pc) {
  deadlock_detector.Lock(type, pc);
}

void DebugMutexUnlock(MutexType type) {
  deadlock_detector.Unlock(type);
}
#endif

}
