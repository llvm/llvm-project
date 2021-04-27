//===-- tsan_sync.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_SYNC_H
#define TSAN_SYNC_H

#include "sanitizer_common/sanitizer_atomic.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_deadlock_detector_interface.h"
#include "tsan_defs.h"
#include "tsan_clock.h"
#include "tsan_shadow.h"
#include "tsan_mutex.h"
#include "tsan_dense_alloc.h"

namespace __tsan {

// These need to match __tsan_mutex_* flags defined in tsan_interface.h.
// See documentation there as well.
enum MutexFlags {
  MutexFlagLinkerInit          = 1 << 0, // __tsan_mutex_linker_init
  MutexFlagWriteReentrant      = 1 << 1, // __tsan_mutex_write_reentrant
  MutexFlagReadReentrant       = 1 << 2, // __tsan_mutex_read_reentrant
  MutexFlagReadLock            = 1 << 3, // __tsan_mutex_read_lock
  MutexFlagTryLock             = 1 << 4, // __tsan_mutex_try_lock
  MutexFlagTryLockFailed       = 1 << 5, // __tsan_mutex_try_lock_failed
  MutexFlagRecursiveLock       = 1 << 6, // __tsan_mutex_recursive_lock
  MutexFlagRecursiveUnlock     = 1 << 7, // __tsan_mutex_recursive_unlock
  MutexFlagNotStatic           = 1 << 8, // __tsan_mutex_not_static

  // The following flags are runtime private.
  // Mutex API misuse was detected, so don't report any more.
  MutexFlagBroken              = 1 << 30,
  // We did not intercept pre lock event, so handle it on post lock.
  MutexFlagDoPreLockOnPostLock = 1 << 29,
  // Must list all mutex creation flags.
  MutexCreationFlagMask        = MutexFlagLinkerInit |
                                 MutexFlagWriteReentrant |
                                 MutexFlagReadReentrant |
                                 MutexFlagNotStatic,
};

//!!! Split for mutex/atomic?
struct SyncVar {
  SyncVar();

  uptr addr;  // overwritten by DenseSlabAlloc freelist
  Mutex mtx;
  Mutex coop;
  StackID creation_stack_id;
  Tid owner_tid; // Set only by exclusive owners.
  FastState last_lock;
  int recursion;
  atomic_uint32_t flags;
  u32 next;  // in MetaMap
  DDMutex dd;
  VectorClock* read_clock; // Used for rw mutexes only.
  VectorClock* clock;

  void Init(ThreadState* thr, uptr pc, uptr addr, bool save_stack);
  void Reset();

  bool IsFlagSet(u32 f) const {
    return atomic_load_relaxed(&flags) & f;
  }

  void SetFlags(u32 f) {
    atomic_store_relaxed(&flags, atomic_load_relaxed(&flags) | f);
  }

  void UpdateFlags(u32 flagz) {
    // Filter out operation flags.
    if (!(flagz & MutexCreationFlagMask))
      return;
    u32 current = atomic_load_relaxed(&flags);
    if (current & MutexCreationFlagMask)
      return;
    // Note: this can be called from MutexPostReadLock which holds only read
    // lock on the SyncVar.
    atomic_store_relaxed(&flags, current | (flagz & MutexCreationFlagMask));
  }
};

// MetaMap allows to map arbitrary user pointers onto various descriptors.
// Currently it maps pointers to heap block descriptors and sync var descs.
// It uses 1/2 direct shadow, see tsan_platform.h.
class MetaMap {
 public:
  MetaMap();

  void AllocBlock(ThreadState *thr, uptr pc, uptr p, uptr sz);
  uptr FreeBlock(Processor *proc, uptr p);
  bool FreeRange(Processor *proc, uptr p, uptr sz);
  void ResetRange(Processor *proc, uptr p, uptr sz);
  void ResetClocks();
  MBlock* GetBlock(uptr p);
  SyncVar* GetSyncOrCreate(ThreadState* thr, uptr pc, uptr addr,
                           bool save_stack) {
    return GetSync(thr, pc, addr, true, save_stack);
  }
  SyncVar* GetSyncIfExists(ThreadState* thr, uptr pc, uptr addr) {
    return GetSync(thr, pc, addr, false, false);
  }

  void MoveMemory(uptr src, uptr dst, uptr sz);

  void OnProcIdle(Processor *proc);

  void GetMemoryStats(uptr* mem_block_mem, uptr* sync_obj_mem) const;

 private:
  static const u32 kFlagMask  = 3u << 30;
  static const u32 kFlagBlock = 1u << 30;
  static const u32 kFlagSync  = 2u << 30;
  typedef DenseSlabAlloc<MBlock, 1 << 18, 1 << 12, kFlagMask> BlockAlloc;
  typedef DenseSlabAlloc<SyncVar, 1 << 20, 1 << 10, kFlagMask> SyncAlloc;
  BlockAlloc block_alloc_;
  SyncAlloc sync_alloc_;

  SyncVar* GetSync(ThreadState* thr, uptr pc, uptr addr, bool create,
                   bool save_stack);
};

}  // namespace __tsan

#endif  // TSAN_SYNC_H
