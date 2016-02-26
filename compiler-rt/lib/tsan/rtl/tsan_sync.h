//===-- tsan_sync.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
#include "tsan_mutex.h"
#include "tsan_dense_alloc.h"

namespace __tsan {

struct SyncVar {
  SyncVar();

  static const int kInvalidTid = -1;

  uptr addr;  // overwritten by DenseSlabAlloc freelist
  Mutex mtx;
  u64 uid;  // Globally unique id.
  u32 creation_stack_id;
  int owner_tid;  // Set only by exclusive owners.
  u64 last_lock;
  int recursion;
  bool is_rw;
  bool is_recursive;
  bool is_broken;
  bool is_linker_init;
  u32 next;  // in MetaMap
  DDMutex dd;
  SyncClock read_clock;  // Used for rw mutexes only.
  // The clock is placed last, so that it is situated on a different cache line
  // with the mtx. This reduces contention for hot sync objects.
  SyncClock clock;

  void Init(ThreadState *thr, uptr pc, uptr addr, u64 uid);
  void Reset(ThreadState *thr);

  u64 GetId() const {
    // 47 lsb is addr, then 14 bits is low part of uid, then 3 zero bits.
    return GetLsb((u64)addr | (uid << 47), 61);
  }
  bool CheckId(u64 uid) const {
    CHECK_EQ(uid, GetLsb(uid, 14));
    return GetLsb(this->uid, 14) == uid;
  }
  static uptr SplitId(u64 id, u64 *uid) {
    *uid = id >> 47;
    return (uptr)GetLsb(id, 47);
  }
};

/* MetaMap allows to map arbitrary user pointers onto various descriptors.
   Currently it maps pointers to heap block descriptors and sync var descs.
   It uses 1/2 direct shadow, see tsan_platform.h.
*/
class MetaMap {
 public:
  MetaMap();

  void AllocBlock(ThreadState *thr, uptr pc, uptr p, uptr sz);
  uptr FreeBlock(ThreadState *thr, uptr pc, uptr p);
  bool FreeRange(ThreadState *thr, uptr pc, uptr p, uptr sz);
  void ResetRange(ThreadState *thr, uptr pc, uptr p, uptr sz);
  MBlock* GetBlock(uptr p);

  SyncVar* GetOrCreateAndLock(ThreadState *thr, uptr pc,
                              uptr addr, bool write_lock);
  SyncVar* GetIfExistsAndLock(uptr addr);

  void MoveMemory(uptr src, uptr dst, uptr sz);

  void OnThreadIdle(ThreadState *thr);

 private:
  static const u32 kFlagMask  = 3u << 30;
  static const u32 kFlagBlock = 1u << 30;
  static const u32 kFlagSync  = 2u << 30;
  typedef DenseSlabAlloc<MBlock, 1<<16, 1<<12> BlockAlloc;
  typedef DenseSlabAlloc<SyncVar, 1<<16, 1<<10> SyncAlloc;
  BlockAlloc block_alloc_;
  SyncAlloc sync_alloc_;
  atomic_uint64_t uid_gen_;

  SyncVar* GetAndLock(ThreadState *thr, uptr pc, uptr addr, bool write_lock,
                      bool create);
};

}  // namespace __tsan

#endif  // TSAN_SYNC_H
