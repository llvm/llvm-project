//===-- tsan_clock.h --------------------------------------------*- C++ -*-===//
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
#ifndef TSAN_CLOCK_H
#define TSAN_CLOCK_H

#include "tsan_defs.h"
#include "tsan_dense_alloc.h"

namespace __tsan {

struct ClockElem {
  u64 epoch  : kClkBits;
  u64 reused : 64 - kClkBits;
};

struct ClockBlock {
  static const uptr kSize = 512;
  static const uptr kTableSize = kSize / sizeof(u32);
  static const uptr kClockCount = kSize / sizeof(ClockElem);

  union {
    u32       table[kTableSize];
    ClockElem clock[kClockCount];
  };

  ClockBlock() {
  }
};

typedef DenseSlabAlloc<ClockBlock, 1<<16, 1<<10> ClockAlloc;
typedef DenseSlabAllocCache ClockCache;

// The clock that lives in sync variables (mutexes, atomics, etc).
class SyncClock {
 public:
  SyncClock();
  ~SyncClock();

  uptr size() const {
    return size_;
  }

  u64 get(unsigned tid) const {
    return elem(tid).epoch;
  }

  void Resize(ClockCache *c, uptr nclk);
  void Reset(ClockCache *c);

  void DebugDump(int(*printf)(const char *s, ...));

 private:
  friend struct ThreadClock;
  static const uptr kDirtyTids = 2;

  unsigned release_store_tid_;
  unsigned release_store_reused_;
  unsigned dirty_tids_[kDirtyTids];
  // tab_ contains indirect pointer to a 512b block using DenseSlabAlloc.
  // If size_ <= 64, then tab_ points to an array with 64 ClockElem's.
  // Otherwise, tab_ points to an array with 128 u32 elements,
  // each pointing to the second-level 512b block with 64 ClockElem's.
  ClockBlock *tab_;
  u32 tab_idx_;
  u32 size_;

  void ResetImpl();
  ClockElem &elem(unsigned tid) const;
};

// The clock that lives in threads.
struct ThreadClock {
 public:
  typedef DenseSlabAllocCache Cache;

  explicit ThreadClock(unsigned tid, unsigned reused = 0);

  u64 get(unsigned tid) const {
    DCHECK_LT(tid, kMaxTidInClock);
    return clk_[tid].epoch;
  }

  void set(ClockCache *c, unsigned tid, u64 v);

  void set(u64 v) {
    DCHECK_GE(v, clk_[tid_].epoch);
    clk_[tid_].epoch = v;
  }

  void tick() {
    clk_[tid_].epoch++;
  }

  uptr size() const {
    return nclk_;
  }

  void acquire(ClockCache *c, const SyncClock *src);
  void release(ClockCache *c, SyncClock *dst) const;
  void acq_rel(ClockCache *c, SyncClock *dst);
  void ReleaseStore(ClockCache *c, SyncClock *dst) const;
  void ResetCached(ClockCache *c);

  void DebugReset();
  void DebugDump(int(*printf)(const char *s, ...));

 private:
  static const uptr kDirtyTids = SyncClock::kDirtyTids;
  const unsigned tid_;
  const unsigned reused_;
  u64 last_acquire_;
  uptr nclk_;
  ClockElem clk_[kMaxTidInClock];

  bool IsAlreadyAcquired(const SyncClock *src) const;
  void UpdateCurrentThread(SyncClock *dst) const;
};

}  // namespace __tsan

#endif  // TSAN_CLOCK_H
