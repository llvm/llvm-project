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

#include "tsan_adaptive_delay.h"

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_allocator_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_errno_codes.h"
#include "tsan_interface.h"
#include "tsan_rtl.h"

namespace __tsan {

namespace {

// =============================================================================
// DelaySpec: Represents a delay configuration parsed from flag strings
// =============================================================================
//
// Delay can be specified as:
//   - "spin=N"     : Spin for up to N cycles (very short delays)
//   - "yield"      : Call sched_yield() once
//   - "sleep_us=N" : Sleep for up to N microseconds

enum class DelayType { Spin, Yield, SleepUs };

struct DelaySpec {
  DelayType type;
  int value;  // spin cycles or sleep_us value; ignored for yield

  // Both estimates below are used internally as a very rough estimate for
  // delay overhead calculation, to cap the overall delay to the
  // adaptive_delay_aggressiveness option. They're not intended to be 100%
  // accurate on any or all architectures/operating systems, or for use in any
  // other contexts.
  //
  // Estimated nanoseconds per spin cycle (volatile loop iteration).
  static constexpr u64 kNsPerSpinCycle = 1;
  // Estimated nanoseconds for a yield (context switch overhead)
  static constexpr u64 kNsPerYield = 500;

  static DelaySpec Parse(const char* str) {
    DelaySpec spec;
    if (internal_strncmp(str, "spin=", 5) == 0) {
      spec.type = DelayType::Spin;
      spec.value = internal_atoll(str + 5);
      if (spec.value <= 0 || spec.value > 10000) {
        Printf(
            "FATAL: Invalid TSAN_OPTIONS spin value '%s'; value must be "
            "between 1 and 10000\n",
            str);
        Die();
      }
    } else if (internal_strcmp(str, "yield") == 0) {
      spec.type = DelayType::Yield;
      spec.value = 0;
    } else if (internal_strncmp(str, "sleep_us=", 9) == 0) {
      spec.type = DelayType::SleepUs;
      spec.value = internal_atoll(str + 9);
      if (spec.value <= 0) {
        Printf(
            "FATAL: Invalid TSAN_OPTIONS sleep_us value '%s'; value must be a "
            "positive integer\n",
            str);
        Die();
      }
    } else {
      Printf("FATAL: Unrecognized delay spec '%s', check TSAN_OPTIONS\n", str);
      Die();
    }
    return spec;
  }

  const char* TypeName() const {
    switch (type) {
      case DelayType::Spin:
        return "spin";
      case DelayType::Yield:
        return "yield";
      case DelayType::SleepUs:
        return "sleep_us";
    }
    return "unknown";
  }
};

}  // namespace

// =============================================================================
// AdaptiveDelayImpl: Time-budget aware delay injection for race exposure
// =============================================================================
//
// This implementation injects delays to expose data races while maintaining a
// configurable overhead target. It uses several strategies:
//
// 1. Time-Budget Controller: Tracks cumulative delays vs wall-clock time
//    and adjusts delay probability to maintain target overhead.
//
// 2. Tiered Delays: Different delay strategies for different op types:
//    - Relaxed atomics: Very rare sampling, tiny spin delays
//    - Sync atomics (acq/rel/seq_cst): Moderate sampling, small usleep
//    - Mutex/CV ops: Higher sampling, larger delays
//    - Thread create/join: Always delay (rare but high value)
//
// 3. Address-based Sampling: Exponential backoff per address to avoid
//    repeatedly delaying hot atomics.

struct AdaptiveDelayImpl {
  ALWAYS_INLINE static AdaptiveDelayState* TLS() {
    return &cur_thread()->adaptive_delay_state;
  }
  ALWAYS_INLINE static unsigned int* GetRandomSeed() {
    return &TLS()->tls_random_seed_;
  }
  ALWAYS_INLINE static void SetRandomSeed(unsigned int seed) {
    TLS()->tls_random_seed_ = seed;
  }

  // The public facing option is adaptive_delay_aggressiveness, which is an
  // opaque value for the user to tune the amount of delay injected into the
  // program. Internally, the implementation maps the aggressiveness to a target
  // percent delay for the overall program runtime. It's not easy to implement
  // a true wall clock delay target (e.g., 25% program wall time slowdown)
  // because 1) spin loops and yield are hard to calculate actual wall time
  // slowness and 2) usleep(N) is often slower than advertised. Thus, we keep
  // the user facing parameter opaque to not under deliver on a promise of
  // percent wall time slowdown.
  struct TimeBudget {
    int target_overhead_pct_;
    Percent target_low_;
    Percent target_high_;

    void Init(int target_pct) {
      target_overhead_pct_ = target_pct;
      target_low_ = Percent::FromPct(
          target_overhead_pct_ >= 5 ? target_overhead_pct_ - 5 : 0);
      target_high_ = Percent::FromPct(target_overhead_pct_ + 5);
    }

    static constexpr u64 BucketDurationNs = 30'000'000'000ULL;

    void RecordDelay(u64 delay_ns) {
      u64 now = NanoTime();
      u64 elapsed_ns = now - TLS()->bucket_start_ns_;

      if (elapsed_ns >= BucketDurationNs) {
        // Shift: old bucket is discarded, new becomes old, start fresh new
        TLS()->delay_buckets_ns_[0] = TLS()->delay_buckets_ns_[1];
        TLS()->delay_buckets_ns_[1] = 0;
        TLS()->bucket_start_ns_ = now;
        TLS()->bucket0_window_ns = BucketDurationNs;
      }

      TLS()->delay_buckets_ns_[1] += delay_ns;
    }

    Percent GetOverheadPercent() {
      u64 now = NanoTime();
      u64 elapsed_ns = now - TLS()->bucket_start_ns_;

      // Need at least 1ms to calculate
      if (elapsed_ns < 1'000'000ULL)
        return Percent::FromPct(0);

      if (elapsed_ns > BucketDurationNs * 2) {
        // Both buckets are stale
        return Percent::FromPct(0);
      } else if (elapsed_ns > BucketDurationNs) {
        // bucket[0] is stale, use only bucket[1] (current bucket)
        u64 total_delay_ns = TLS()->delay_buckets_ns_[1];
        return Percent::FromRatio(total_delay_ns, elapsed_ns);
      } else {
        u64 total_delay_ns =
            TLS()->delay_buckets_ns_[0] + TLS()->delay_buckets_ns_[1];
        u64 window_ns = TLS()->bucket0_window_ns + elapsed_ns;
        return Percent::FromRatio(total_delay_ns, window_ns);
      }
    }

    bool ShouldDelay() {
      Percent ratio = GetOverheadPercent();

      if (ratio < target_low_)
        return true;
      if (ratio > target_high_)
        return false;

      // Linear interpolation: at target_low -> 100%, at target_high -> 0%
      Percent prob = (target_high_ - ratio) / (target_high_ - target_low_);
      return prob.RandomCheck(GetRandomSeed());
    }
  };

  // Address Sampler with Exponential Backoff
  struct AddressSampler {
    static constexpr u64 TABLE_SIZE = 2048;
    struct Entry {
      atomic_uintptr_t addr_;
      atomic_uint32_t count_;
    };
    Entry table_[TABLE_SIZE];
    static constexpr u32 ExponentialBackoffCap = 64;

    void Init() {
      for (u64 i = 0; i < TABLE_SIZE; ++i) {
        atomic_store(&table_[i].addr_, 0, memory_order_relaxed);
        atomic_store(&table_[i].count_, 0, memory_order_relaxed);
      }
    }

    static ALWAYS_INLINE u64 splitmix64(u64 x) {
      x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
      x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
      x = x ^ (x >> 31);
      return x;
    }

    // Uses exponential backoff: delay on 1st, 2nd, 4th, 8th, 16th, ...
    bool ShouldDelayAddr(uptr addr) {
      u64 idx = splitmix64(addr >> 3) & (TABLE_SIZE - 1);
      Entry& e = table_[idx];

      // This function is not thread safe.
      // If two threads access the same hashed entry in parallel,
      // worst case, we may end up returning true too often. This is
      // acceptable...instead of full locking.

      uptr stored_addr = atomic_load(&e.addr_, memory_order_relaxed);
      if (stored_addr != addr) {
        // Hash Collision - reset
        atomic_store(&e.addr_, addr, memory_order_relaxed);
        atomic_store(&e.count_, 1, memory_order_relaxed);
        return true;
      }

      u32 count = atomic_fetch_add(&e.count_, 1, memory_order_relaxed) + 1;

      if ((count & (count - 1)) == 0 && count <= ExponentialBackoffCap)
        return true;
      return false;
    }
  };

  TimeBudget budget_;
  AddressSampler sampler_;

  int relaxed_sample_rate_;
  int sync_atomic_sample_rate_;
  int mutex_sample_rate_;
  DelaySpec atomic_delay_;
  DelaySpec sync_delay_;

  void Init() { InitTls(); }

  void InitTls() {
    TLS()->bucket_start_ns_ = NanoTime();
    TLS()->delay_buckets_ns_[0] = 0;
    TLS()->delay_buckets_ns_[1] = 0;
    TLS()->bucket0_window_ns = 0;

    SetRandomSeed(NanoTime());
    TLS()->tls_initialized_ = true;
  }

  bool IsTlsInitialized() const { return TLS()->tls_initialized_; }

  AdaptiveDelayImpl() {
    relaxed_sample_rate_ = flags()->adaptive_delay_relaxed_sample_rate;
    sync_atomic_sample_rate_ = flags()->adaptive_delay_sync_atomic_sample_rate;
    mutex_sample_rate_ = flags()->adaptive_delay_mutex_sample_rate;
    atomic_delay_ = DelaySpec::Parse(flags()->adaptive_delay_max_atomic);
    sync_delay_ = DelaySpec::Parse(flags()->adaptive_delay_max_sync);

    int delay_aggressiveness = flags()->adaptive_delay_aggressiveness;
    if (delay_aggressiveness < 1)
      delay_aggressiveness = 1;

    budget_.Init(delay_aggressiveness);
    sampler_.Init();

    VPrintf(1, "INFO: ThreadSanitizer AdaptiveDelay initialized\n");
    VPrintf(1, "  Delay aggressiveness: %d\n", delay_aggressiveness);
    VPrintf(1, "  Relaxed atomic sample rate: 1/%d\n", relaxed_sample_rate_);
    VPrintf(1, "  Sync atomic sample rate: 1/%d\n", sync_atomic_sample_rate_);
    VPrintf(1, "  Mutex sample rate: 1/%d\n", mutex_sample_rate_);
    VPrintf(1, "  Atomic delay: %s=%d\n", atomic_delay_.TypeName(),
            atomic_delay_.value);
    VPrintf(1, "  Sync delay: %s=%d\n", sync_delay_.TypeName(),
            sync_delay_.value);
  }

  void DoSpinDelay(int iters) {
    volatile int v = 0;
    for (int i = 0; i < iters; ++i) v = i;
    (void)v;
    budget_.RecordDelay(iters * DelaySpec::kNsPerSpinCycle);
  }

  void DoYieldDelay() {
    internal_sched_yield();
    budget_.RecordDelay(DelaySpec::kNsPerYield);
  }

  void DoSleepUsDelay(int max_us) {
    // Use two Rand() calls to get full 32-bit range for larger sleep values
    u32 rnd = ((u32)Rand(GetRandomSeed()) << 16) | Rand(GetRandomSeed());
    int delay_us = 1 + (rnd % max_us);
    internal_usleep(delay_us);
    budget_.RecordDelay(delay_us * 1000ULL);
  }

  void ExecuteDelay(const DelaySpec& spec) {
    switch (spec.type) {
      case DelayType::Spin: {
        int iters = 1 + (Rand(GetRandomSeed()) % spec.value);
        DoSpinDelay(iters);
        break;
      }
      case DelayType::Yield:
        DoYieldDelay();
        break;
      case DelayType::SleepUs:
        DoSleepUsDelay(spec.value);
        break;
    }
  }

  void AtomicRelaxedOpDelay() {
    if ((Rand(GetRandomSeed()) % relaxed_sample_rate_) != 0)
      return;
    if (!budget_.ShouldDelay())
      return;

    int iters = 10 + (Rand(GetRandomSeed()) % 10);
    DoSpinDelay(iters);
  }

  void AtomicSyncOpDelay(uptr* addr) {
    if ((Rand(GetRandomSeed()) % sync_atomic_sample_rate_) != 0)
      return;
    if (!budget_.ShouldDelay())
      return;

    if (addr && !sampler_.ShouldDelayAddr(*addr))
      return;

    ExecuteDelay(atomic_delay_);
  }

  void AtomicOpFence(int mo) {
    CHECK(IsTlsInitialized());

    if (mo < mo_acquire)
      AtomicRelaxedOpDelay();
    else
      AtomicSyncOpDelay(nullptr);
  }

  void AtomicOpAddr(uptr addr, int mo) {
    CHECK(IsTlsInitialized());

    if (mo < mo_acquire)
      AtomicRelaxedOpDelay();
    else
      AtomicSyncOpDelay(&addr);
  }

  void UnsampledDelay() {
    CHECK(IsTlsInitialized());

    if (!budget_.ShouldDelay())
      return;

    ExecuteDelay(sync_delay_);
  }

  void SyncOp() {
    CHECK(IsTlsInitialized());

    if ((Rand(GetRandomSeed()) % mutex_sample_rate_) != 0)
      return;
    if (!budget_.ShouldDelay())
      return;

    ExecuteDelay(sync_delay_);
  }

  void BeforeChildThreadRuns() {
    InitTls();
    UnsampledDelay();
  }

  void AfterThreadCreation() { UnsampledDelay(); }
};

AdaptiveDelayImpl& GetImpl() {
  static AdaptiveDelayImpl impl;
  return impl;
}

bool AdaptiveDelay::is_adaptive_delay_enabled;

void AdaptiveDelay::InitImpl() {
  AdaptiveDelay::is_adaptive_delay_enabled = flags()->enable_adaptive_delay;
  if (!AdaptiveDelay::is_adaptive_delay_enabled)
    return;

  GetImpl().Init();
}

void AdaptiveDelay::SyncOpImpl() { GetImpl().SyncOp(); }
void AdaptiveDelay::AtomicOpFenceImpl(int mo) { GetImpl().AtomicOpFence(mo); }
void AdaptiveDelay::AtomicOpAddrImpl(__sanitizer::uptr addr, int mo) {
  GetImpl().AtomicOpAddr(addr, mo);
}
void AdaptiveDelay::AfterThreadCreationImpl() {
  GetImpl().AfterThreadCreation();
}
void AdaptiveDelay::BeforeChildThreadRunsImpl() {
  GetImpl().BeforeChildThreadRuns();
}

}  // namespace __tsan
