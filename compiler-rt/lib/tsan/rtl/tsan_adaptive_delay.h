//===-- tsan_adaptive_delay.h -----------------------------------*- C++ -*-===//
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

#ifndef TSAN_ADAPTIVE_DELAY_H
#define TSAN_ADAPTIVE_DELAY_H

#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __tsan {

// AdaptiveDelay injects delays at synchronization points, atomic operations,
// and thread lifecycle events to increase the likelihood of exposing data
// races. The delay injection is controlled by an approximate time budget to
// maintain a configurable overhead target.
//
// SyncOp() delays non-atomic synchronization points (those with clear
// happens-before relationships):
//  - Acquire operations like locking a mutex delays before the mutex is locked.
//  - Release operations like unlocking a mutex delays after the mutex is
//  unlocked
// These are more likely to expose interesting (rare) thread interleavings.
// For example, delaying a thread that unlocks a mutex from running to allow
// newly woken thread to execute before the unlocking thread would normally
// execute.
//
// TODO:
//  - Move the adaptive delay implementation into sanitizer_common so that
//    ASAN can also leverage it in pthread_* interceptors
//  - Integrate into other interceptors like libdispatch.
struct AdaptiveDelay {
  ALWAYS_INLINE static void Init() { InitImpl(); }

  ALWAYS_INLINE static void SyncOp() {
    if (!is_adaptive_delay_enabled)
      return;
    SyncOpImpl();
  }

  ALWAYS_INLINE static void AtomicOpFence(int mo) {
    if (!is_adaptive_delay_enabled)
      return;
    AtomicOpFenceImpl(mo);
  }

  ALWAYS_INLINE static void AtomicOpAddr(__sanitizer::uptr addr, int mo) {
    if (!is_adaptive_delay_enabled)
      return;
    AtomicOpAddrImpl(addr, mo);
  }

  ALWAYS_INLINE static void AfterThreadCreation() {
    if (!is_adaptive_delay_enabled)
      return;
    AfterThreadCreationImpl();
  }

  ALWAYS_INLINE static void BeforeChildThreadRuns() {
    if (!is_adaptive_delay_enabled)
      return;
    BeforeChildThreadRunsImpl();
  }

 private:
  static void InitImpl();

  static void SyncOpImpl();

  static void AtomicOpFenceImpl(int mo);
  static void AtomicOpAddrImpl(__sanitizer::uptr addr, int mo);

  static void AfterThreadCreationImpl();
  static void BeforeChildThreadRunsImpl();

  static bool is_adaptive_delay_enabled;
};

// The runtime defines cur_thread() to retrieve TLS thread state, and it
// takes care of platform specific implementation details. The AdaptiveDelay
// implementation stores per-thread data in this struct, which is embedded
// in cur_thread().
struct AdaptiveDelayState {
  // For the adaptive delay implementation
  // Sliding window delay tracking: 2 buckets of 30 seconds each
  u64 delay_buckets_ns_[2];  // [0] = older 30s, [1] = newer 30s
  u64 bucket_start_ns_;      // When current bucket (index 1) started
  u64 bucket0_window_ns;  // 0ns before the first bucket has rolled, and set to
                          // the bucket window time after This handles the case
                          // where, before the program has ran one bucket window
                          // duration, we should not include the previous bucket
                          // duration in the overhead percent calculation.
  unsigned int tls_random_seed_;
  bool tls_initialized_;
};

// Fixed-point arithmetic type that mimics floating point operations
class Percent {
  using u32 = __sanitizer::u32;
  using u64 = __sanitizer::u64;

  u32 bp_{};  // basis points (0-10000 represents 0.0-1.0)
  bool is_valid_{};

  static constexpr u32 kBasisPointsPerUnit = 10000;

  Percent(u32 bp, bool is_valid) : bp_(bp), is_valid_(is_valid) {}

 public:
  Percent() = default;
  Percent(const Percent&) = default;
  Percent& operator=(const Percent&) = default;
  Percent(Percent&&) = default;
  Percent& operator=(Percent&&) = default;

  static Percent FromPct(u32 pct) { return Percent{pct * 100, true}; }
  static Percent FromRatio(u64 numerator, u64 denominator) {
    if (denominator == 0)
      return Percent{0, false};
    // Avoid overflow: scale down if needed
    if (numerator > UINT64_MAX / kBasisPointsPerUnit) {
      return Percent{(u32)((numerator / denominator) * kBasisPointsPerUnit),
                     true};
    }
    return Percent{(u32)((numerator * kBasisPointsPerUnit) / denominator),
                   true};
  }

  bool IsValid() const { return is_valid_; }

  // Returns true with probability equal to the percentage.
  bool RandomCheck(u32* seed) const {
    return (Rand(seed) % kBasisPointsPerUnit) < bp_;
  }

  int GetPct() const { return bp_ / 100; }
  int GetBasisPoints() const { return bp_; }

  bool operator==(const Percent& other) const { return bp_ == other.bp_; }
  bool operator!=(const Percent& other) const { return bp_ != other.bp_; }
  bool operator<(const Percent& other) const { return bp_ < other.bp_; }
  bool operator>(const Percent& other) const { return bp_ > other.bp_; }
  bool operator<=(const Percent& other) const { return bp_ <= other.bp_; }
  bool operator>=(const Percent& other) const { return bp_ >= other.bp_; }

  Percent operator-(const Percent& other) const {
    if (!is_valid_ || !other.is_valid_)
      return Percent{0, false};
    if (bp_ < other.bp_)
      return Percent{0, false};
    return Percent{bp_ - other.bp_, true};
  }

  Percent operator/(const Percent& other) const {
    if (!is_valid_ || !other.is_valid_)
      return Percent{0, false};
    if (other.bp_ == 0)
      return Percent{0, false};
    return Percent{(bp_ * kBasisPointsPerUnit) / other.bp_, true};
  }
};

}  // namespace __tsan

#endif  // TSAN_ADAPTIVE_DELAY_H
