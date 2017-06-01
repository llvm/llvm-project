//===-- tsan_clock.cc -----------------------------------------------------===//
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
#include "tsan_clock.h"
#include "tsan_rtl.h"
#include "sanitizer_common/sanitizer_placement_new.h"

// SyncClock and ThreadClock implement vector clocks for sync variables
// (mutexes, atomic variables, file descriptors, etc) and threads, respectively.
// ThreadClock contains fixed-size vector clock for maximum number of threads.
// SyncClock contains growable vector clock for currently necessary number of
// threads.
// Together they implement very simple model of operations, namely:
//
//   void ThreadClock::acquire(const SyncClock *src) {
//     for (int i = 0; i < kMaxThreads; i++)
//       clock[i] = max(clock[i], src->clock[i]);
//   }
//
//   void ThreadClock::release(SyncClock *dst) const {
//     for (int i = 0; i < kMaxThreads; i++)
//       dst->clock[i] = max(dst->clock[i], clock[i]);
//   }
//
//   void ThreadClock::ReleaseStore(SyncClock *dst) const {
//     for (int i = 0; i < kMaxThreads; i++)
//       dst->clock[i] = clock[i];
//   }
//
//   void ThreadClock::acq_rel(SyncClock *dst) {
//     acquire(dst);
//     release(dst);
//   }
//
// Conformance to this model is extensively verified in tsan_clock_test.cc.
// However, the implementation is significantly more complex. The complexity
// allows to implement important classes of use cases in O(1) instead of O(N).
//
// The use cases are:
// 1. Singleton/once atomic that has a single release-store operation followed
//    by zillions of acquire-loads (the acquire-load is O(1)).
// 2. Thread-local mutex (both lock and unlock can be O(1)).
// 3. Leaf mutex (unlock is O(1)).
// 4. A mutex shared by 2 threads (both lock and unlock can be O(1)).
// 5. An atomic with a single writer (writes can be O(1)).
// The implementation dynamically adopts to workload. So if an atomic is in
// read-only phase, these reads will be O(1); if it later switches to read/write
// phase, the implementation will correctly handle that by switching to O(N).
//
// Thread-safety note: all const operations on SyncClock's are conducted under
// a shared lock; all non-const operations on SyncClock's are conducted under
// an exclusive lock; ThreadClock's are private to respective threads and so
// do not need any protection.
//
// Description of ThreadClock state:
// clk_ - fixed size vector clock.
// nclk_ - effective size of the vector clock (the rest is zeros).
// tid_ - index of the thread associated with he clock ("current thread").
// last_acquire_ - current thread time when it acquired something from
//   other threads.
//
// Description of SyncClock state:
// clk_ - variable size vector clock, low kClkBits hold timestamp,
//   the remaining bits hold "acquired" flag (the actual value is thread's
//   reused counter);
//   if acquried == thr->reused_, then the respective thread has already
//   acquired this clock (except possibly dirty_tids_).
// dirty_tids_ - holds up to two indeces in the vector clock that other threads
//   need to acquire regardless of "acquired" flag value;
// release_store_tid_ - denotes that the clock state is a result of
//   release-store operation by the thread with release_store_tid_ index.
// release_store_reused_ - reuse count of release_store_tid_.

// We don't have ThreadState in these methods, so this is an ugly hack that
// works only in C++.
#if !SANITIZER_GO
# define CPP_STAT_INC(typ) StatInc(cur_thread(), typ)
#else
# define CPP_STAT_INC(typ) (void)0
#endif

namespace __tsan {

ThreadClock::ThreadClock(unsigned tid, unsigned reused)
    : tid_(tid)
    , reused_(reused + 1) {  // 0 has special meaning
  CHECK_LT(tid, kMaxTidInClock);
  CHECK_EQ(reused_, ((u64)reused_ << kClkBits) >> kClkBits);
  nclk_ = tid_ + 1;
  last_acquire_ = 0;
  internal_memset(clk_, 0, sizeof(clk_));
  clk_[tid_].reused = reused_;
}

void ThreadClock::acquire(ClockCache *c, const SyncClock *src) {
  DCHECK_LE(nclk_, kMaxTid);
  DCHECK_LE(src->size_, kMaxTid);
  CPP_STAT_INC(StatClockAcquire);

  // Check if it's empty -> no need to do anything.
  const uptr nclk = src->size_;
  if (nclk == 0) {
    CPP_STAT_INC(StatClockAcquireEmpty);
    return;
  }

  // Check if we've already acquired src after the last release operation on src
  bool acquired = false;
  if (nclk > tid_) {
    CPP_STAT_INC(StatClockAcquireLarge);
    if (src->elem(tid_).reused == reused_) {
      CPP_STAT_INC(StatClockAcquireRepeat);
      for (unsigned i = 0; i < kDirtyTids; i++) {
        unsigned tid = src->dirty_tids_[i];
        if (tid != kInvalidTid) {
          u64 epoch = src->elem(tid).epoch;
          if (clk_[tid].epoch < epoch) {
            clk_[tid].epoch = epoch;
            acquired = true;
          }
        }
      }
      if (acquired) {
        CPP_STAT_INC(StatClockAcquiredSomething);
        last_acquire_ = clk_[tid_].epoch;
      }
      return;
    }
  }

  // O(N) acquire.
  CPP_STAT_INC(StatClockAcquireFull);
  nclk_ = max(nclk_, nclk);
  for (uptr i = 0; i < nclk; i++) {
    u64 epoch = src->elem(i).epoch;
    if (clk_[i].epoch < epoch) {
      clk_[i].epoch = epoch;
      acquired = true;
    }
  }

  // Remember that this thread has acquired this clock.
  if (nclk > tid_)
    src->elem(tid_).reused = reused_;

  if (acquired) {
    CPP_STAT_INC(StatClockAcquiredSomething);
    last_acquire_ = clk_[tid_].epoch;
  }
}

void ThreadClock::release(ClockCache *c, SyncClock *dst) const {
  DCHECK_LE(nclk_, kMaxTid);
  DCHECK_LE(dst->size_, kMaxTid);

  if (dst->size_ == 0) {
    // ReleaseStore will correctly set release_store_tid_,
    // which can be important for future operations.
    ReleaseStore(c, dst);
    return;
  }

  CPP_STAT_INC(StatClockRelease);
  // Check if we need to resize dst.
  if (dst->size_ < nclk_)
    dst->Resize(c, nclk_);

  // Check if we had not acquired anything from other threads
  // since the last release on dst. If so, we need to update
  // only dst->elem(tid_).
  if (dst->elem(tid_).epoch > last_acquire_) {
    UpdateCurrentThread(dst);
    if (dst->release_store_tid_ != tid_ ||
        dst->release_store_reused_ != reused_)
      dst->release_store_tid_ = kInvalidTid;
    return;
  }

  // O(N) release.
  CPP_STAT_INC(StatClockReleaseFull);
  // First, remember whether we've acquired dst.
  bool acquired = IsAlreadyAcquired(dst);
  if (acquired)
    CPP_STAT_INC(StatClockReleaseAcquired);
  // Update dst->clk_.
  for (uptr i = 0; i < nclk_; i++) {
    ClockElem &ce = dst->elem(i);
    ce.epoch = max(ce.epoch, clk_[i].epoch);
    ce.reused = 0;
  }
  // Clear 'acquired' flag in the remaining elements.
  if (nclk_ < dst->size_)
    CPP_STAT_INC(StatClockReleaseClearTail);
  for (uptr i = nclk_; i < dst->size_; i++)
    dst->elem(i).reused = 0;
  for (unsigned i = 0; i < kDirtyTids; i++)
    dst->dirty_tids_[i] = kInvalidTid;
  dst->release_store_tid_ = kInvalidTid;
  dst->release_store_reused_ = 0;
  // If we've acquired dst, remember this fact,
  // so that we don't need to acquire it on next acquire.
  if (acquired)
    dst->elem(tid_).reused = reused_;
}

void ThreadClock::ReleaseStore(ClockCache *c, SyncClock *dst) const {
  DCHECK_LE(nclk_, kMaxTid);
  DCHECK_LE(dst->size_, kMaxTid);
  CPP_STAT_INC(StatClockStore);

  // Check if we need to resize dst.
  if (dst->size_ < nclk_)
    dst->Resize(c, nclk_);

  if (dst->release_store_tid_ == tid_ &&
      dst->release_store_reused_ == reused_ &&
      dst->elem(tid_).epoch > last_acquire_) {
    CPP_STAT_INC(StatClockStoreFast);
    UpdateCurrentThread(dst);
    return;
  }

  // O(N) release-store.
  CPP_STAT_INC(StatClockStoreFull);
  for (uptr i = 0; i < nclk_; i++) {
    ClockElem &ce = dst->elem(i);
    ce.epoch = clk_[i].epoch;
    ce.reused = 0;
  }
  // Clear the tail of dst->clk_.
  if (nclk_ < dst->size_) {
    for (uptr i = nclk_; i < dst->size_; i++) {
      ClockElem &ce = dst->elem(i);
      ce.epoch = 0;
      ce.reused = 0;
    }
    CPP_STAT_INC(StatClockStoreTail);
  }
  for (unsigned i = 0; i < kDirtyTids; i++)
    dst->dirty_tids_[i] = kInvalidTid;
  dst->release_store_tid_ = tid_;
  dst->release_store_reused_ = reused_;
  // Rememeber that we don't need to acquire it in future.
  dst->elem(tid_).reused = reused_;
}

void ThreadClock::acq_rel(ClockCache *c, SyncClock *dst) {
  CPP_STAT_INC(StatClockAcquireRelease);
  acquire(c, dst);
  ReleaseStore(c, dst);
}

// Updates only single element related to the current thread in dst->clk_.
void ThreadClock::UpdateCurrentThread(SyncClock *dst) const {
  // Update the threads time, but preserve 'acquired' flag.
  dst->elem(tid_).epoch = clk_[tid_].epoch;

  for (unsigned i = 0; i < kDirtyTids; i++) {
    if (dst->dirty_tids_[i] == tid_) {
      CPP_STAT_INC(StatClockReleaseFast1);
      return;
    }
    if (dst->dirty_tids_[i] == kInvalidTid) {
      CPP_STAT_INC(StatClockReleaseFast2);
      dst->dirty_tids_[i] = tid_;
      return;
    }
  }
  // Reset all 'acquired' flags, O(N).
  CPP_STAT_INC(StatClockReleaseSlow);
  for (uptr i = 0; i < dst->size_; i++)
    dst->elem(i).reused = 0;
  for (unsigned i = 0; i < kDirtyTids; i++)
    dst->dirty_tids_[i] = kInvalidTid;
}

// Checks whether the current threads has already acquired src.
bool ThreadClock::IsAlreadyAcquired(const SyncClock *src) const {
  if (src->elem(tid_).reused != reused_)
    return false;
  for (unsigned i = 0; i < kDirtyTids; i++) {
    unsigned tid = src->dirty_tids_[i];
    if (tid != kInvalidTid) {
      if (clk_[tid].epoch < src->elem(tid).epoch)
        return false;
    }
  }
  return true;
}

void SyncClock::Resize(ClockCache *c, uptr nclk) {
  CPP_STAT_INC(StatClockReleaseResize);
  if (RoundUpTo(nclk, ClockBlock::kClockCount) <=
      RoundUpTo(size_, ClockBlock::kClockCount)) {
    // Growing within the same block.
    // Memory is already allocated, just increase the size.
    size_ = nclk;
    return;
  }
  if (nclk <= ClockBlock::kClockCount) {
    // Grow from 0 to one-level table.
    CHECK_EQ(size_, 0);
    CHECK_EQ(tab_, 0);
    CHECK_EQ(tab_idx_, 0);
    size_ = nclk;
    tab_idx_ = ctx->clock_alloc.Alloc(c);
    tab_ = ctx->clock_alloc.Map(tab_idx_);
    internal_memset(tab_, 0, sizeof(*tab_));
    return;
  }
  // Growing two-level table.
  if (size_ == 0) {
    // Allocate first level table.
    tab_idx_ = ctx->clock_alloc.Alloc(c);
    tab_ = ctx->clock_alloc.Map(tab_idx_);
    internal_memset(tab_, 0, sizeof(*tab_));
  } else if (size_ <= ClockBlock::kClockCount) {
    // Transform one-level table to two-level table.
    u32 old = tab_idx_;
    tab_idx_ = ctx->clock_alloc.Alloc(c);
    tab_ = ctx->clock_alloc.Map(tab_idx_);
    internal_memset(tab_, 0, sizeof(*tab_));
    tab_->table[0] = old;
  }
  // At this point we have first level table allocated.
  // Add second level tables as necessary.
  for (uptr i = RoundUpTo(size_, ClockBlock::kClockCount);
      i < nclk; i += ClockBlock::kClockCount) {
    u32 idx = ctx->clock_alloc.Alloc(c);
    ClockBlock *cb = ctx->clock_alloc.Map(idx);
    internal_memset(cb, 0, sizeof(*cb));
    CHECK_EQ(tab_->table[i/ClockBlock::kClockCount], 0);
    tab_->table[i/ClockBlock::kClockCount] = idx;
  }
  size_ = nclk;
}

// Sets a single element in the vector clock.
// This function is called only from weird places like AcquireGlobal.
void ThreadClock::set(unsigned tid, u64 v) {
  DCHECK_LT(tid, kMaxTid);
  DCHECK_GE(v, clk_[tid].epoch);
  clk_[tid].epoch = v;
  if (nclk_ <= tid)
    nclk_ = tid + 1;
  last_acquire_ = clk_[tid_].epoch;
}

void ThreadClock::DebugDump(int(*printf)(const char *s, ...)) {
  printf("clock=[");
  for (uptr i = 0; i < nclk_; i++)
    printf("%s%llu", i == 0 ? "" : ",", clk_[i].epoch);
  printf("] reused=[");
  for (uptr i = 0; i < nclk_; i++)
    printf("%s%llu", i == 0 ? "" : ",", clk_[i].reused);
  printf("] tid=%u/%u last_acq=%llu",
      tid_, reused_, last_acquire_);
}

SyncClock::SyncClock()
    : release_store_tid_(kInvalidTid)
    , release_store_reused_()
    , tab_()
    , tab_idx_()
    , size_() {
  for (uptr i = 0; i < kDirtyTids; i++)
    dirty_tids_[i] = kInvalidTid;
}

SyncClock::~SyncClock() {
  // Reset must be called before dtor.
  CHECK_EQ(size_, 0);
  CHECK_EQ(tab_, 0);
  CHECK_EQ(tab_idx_, 0);
}

void SyncClock::Reset(ClockCache *c) {
  if (size_ == 0) {
    // nothing
  } else if (size_ <= ClockBlock::kClockCount) {
    // One-level table.
    ctx->clock_alloc.Free(c, tab_idx_);
  } else {
    // Two-level table.
    for (uptr i = 0; i < size_; i += ClockBlock::kClockCount)
      ctx->clock_alloc.Free(c, tab_->table[i / ClockBlock::kClockCount]);
    ctx->clock_alloc.Free(c, tab_idx_);
  }
  tab_ = 0;
  tab_idx_ = 0;
  size_ = 0;
  release_store_tid_ = kInvalidTid;
  release_store_reused_ = 0;
  for (uptr i = 0; i < kDirtyTids; i++)
    dirty_tids_[i] = kInvalidTid;
}

ClockElem &SyncClock::elem(unsigned tid) const {
  DCHECK_LT(tid, size_);
  if (size_ <= ClockBlock::kClockCount)
    return tab_->clock[tid];
  u32 idx = tab_->table[tid / ClockBlock::kClockCount];
  ClockBlock *cb = ctx->clock_alloc.Map(idx);
  return cb->clock[tid % ClockBlock::kClockCount];
}

void SyncClock::DebugDump(int(*printf)(const char *s, ...)) {
  printf("clock=[");
  for (uptr i = 0; i < size_; i++)
    printf("%s%llu", i == 0 ? "" : ",", elem(i).epoch);
  printf("] reused=[");
  for (uptr i = 0; i < size_; i++)
    printf("%s%llu", i == 0 ? "" : ",", elem(i).reused);
  printf("] release_store_tid=%d/%d dirty_tids=%d/%d",
      release_store_tid_, release_store_reused_,
      dirty_tids_[0], dirty_tids_[1]);
}
}  // namespace __tsan
