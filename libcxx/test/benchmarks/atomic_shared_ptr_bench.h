//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_BENCHMARK_ATOMIC_SHARED_PTR_BENCH_H
#define TEST_BENCHMARK_ATOMIC_SHARED_PTR_BENCH_H

// libc++ lock-free vs lock-based microbenchmarks for
// std::atomic<std::shared_ptr<T>>. Names match so compare-benchmarks can pair
// the two binaries:
//
//   lock-free (default on x86_64/cx16 and AArch64/LSE):
//     atomic_shared_ptr.bench.cpp
//   lock-based (forced, same machine):
//     atomic_shared_ptr_lock_based.bench.cpp
//
// Normalize on the same machine: divide each shared_ptr result by the matching
// std::atomic<uint64_t> bench in the same run. Absolute ns/op are host-specific;
// the ratio is the number to report.
//
// Driver: libcxx/test/benchmarks/atomic_shared_ptr_bench.runner.py

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <vector>

#include "benchmark/benchmark.h"
#include "test_macros.h"

#ifdef _LIBCPP_HAS_LOCKFREE_ATOMIC_SHARED_PTR
#  if _LIBCPP_HAS_LOCKFREE_ATOMIC_SHARED_PTR
#    define ATOMIC_SP_IMPL_LABEL "libcxx_lock_free"
#  else
#    define ATOMIC_SP_IMPL_LABEL "libcxx_lock_based"
#  endif
#else
#  define ATOMIC_SP_IMPL_LABEL "libcxx_lock_based"
#endif

inline void annotate_atomic_sp(benchmark::State& state) { state.SetLabel(ATOMIC_SP_IMPL_LABEL); }

// Contended thread counts. Override at process start (before BENCHMARK
// registration) with ATOMIC_SP_BENCH_THREADS=2,4,6,8. Default: 2,4,8.
// Google Benchmark has no --threads CLI that overrides ->Threads(N).
inline std::vector<int> atomic_sp_thread_counts() {
  std::vector<int> counts;
  const char* env = std::getenv("ATOMIC_SP_BENCH_THREADS");
  const char* p   = (env != nullptr && env[0] != '\0') ? env : "2,4,8";
  while (*p != '\0') {
    while (*p == ' ' || *p == '\t' || *p == ',')
      ++p;
    if (*p == '\0')
      break;
    char* end    = nullptr;
    const long v = std::strtol(p, &end, 10);
    if (end == p)
      break;
    if (v >= 1 && v <= 1024) {
      const int t = static_cast<int>(v);
      bool seen   = false;
      for (int existing : counts) {
        if (existing == t) {
          seen = true;
          break;
        }
      }
      if (!seen)
        counts.push_back(t);
    }
    p = end;
  }
  if (counts.empty()) {
    counts.push_back(2);
    counts.push_back(4);
    counts.push_back(8);
  }
  return counts;
}

inline void apply_contended_threads(benchmark::Benchmark* b) {
  for (int t : atomic_sp_thread_counts())
    b->Threads(t);
  b->UseRealTime();
}

// Baseline: lock-free scalar CAS/load/store on the same binary. Use these as
// the denominator when reporting normalized numbers.

static TEST_ALIGN_BENCHMARK void bm_atomic_u64_load_uncontended(benchmark::State& state) {
  annotate_atomic_sp(state);
  std::atomic<std::uint64_t> atom{42};
  for (auto _ : state) {
    auto snap = atom.load();
    benchmark::DoNotOptimize(snap);
  }
}
BENCHMARK(bm_atomic_u64_load_uncontended)->Name("std::atomic<uint64_t>::load() (uncontended)");

static TEST_ALIGN_BENCHMARK void bm_atomic_u64_store_uncontended(benchmark::State& state) {
  annotate_atomic_sp(state);
  std::atomic<std::uint64_t> atom{0};
  for (auto _ : state) {
    atom.store(7);
    // Without this the store is not observed and -O3 DSE's the whole loop
    // (the previous 0.000 ns / 1e12-iteration row).
    benchmark::DoNotOptimize(atom);
  }
}
BENCHMARK(bm_atomic_u64_store_uncontended)->Name("std::atomic<uint64_t>::store() (uncontended)");

static TEST_ALIGN_BENCHMARK void bm_atomic_u64_cas_uncontended(benchmark::State& state) {
  annotate_atomic_sp(state);
  std::atomic<std::uint64_t> atom{1};
  for (auto _ : state) {
    std::uint64_t expected = atom.load(std::memory_order_relaxed);
    (void)atom.compare_exchange_strong(expected, expected ^ 1);
  }
}
BENCHMARK(bm_atomic_u64_cas_uncontended)->Name("std::atomic<uint64_t>::compare_exchange_strong() (uncontended)");

static TEST_ALIGN_BENCHMARK void bm_atomic_u64_load_contended(benchmark::State& state) {
  annotate_atomic_sp(state);
  static std::atomic<std::uint64_t> atom{42};
  if (state.thread_index() == 0)
    atom.store(42);
  for (auto _ : state) {
    auto snap = atom.load();
    benchmark::DoNotOptimize(snap);
  }
}
BENCHMARK(bm_atomic_u64_load_contended)
    ->Name("std::atomic<uint64_t>::load() (contended)")
    ->Apply(apply_contended_threads);

static TEST_ALIGN_BENCHMARK void bm_atomic_u64_cas_contended(benchmark::State& state) {
  annotate_atomic_sp(state);
  static std::atomic<std::uint64_t> atom{1};
  if (state.thread_index() == 0)
    atom.store(1);
  for (auto _ : state) {
    std::uint64_t expected = atom.load(std::memory_order_relaxed);
    (void)atom.compare_exchange_strong(expected, expected ^ 1);
  }
}
BENCHMARK(bm_atomic_u64_cas_contended)
    ->Name("std::atomic<uint64_t>::compare_exchange_strong() (contended)")
    ->Apply(apply_contended_threads);

// std::atomic<shared_ptr<T>>. Uncontended benches use a per-thread object.
// Contended benches share one atomic (function-local static) so Threads(N)
// is real contention, not N independent copies.
//
// Google Benchmark registers each ->Threads(K) as a separate invocation of
// the same C++ function in one process. The static is initialized once;
// without an explicit reset, CAS/store leftovers leak from threads:2 into
// threads:4/8 (and across KeepRunning calibration). Thread 0 resets before
// the timed loop; the loop begin is a barrier, so other threads wait.

static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_load_uncontended(benchmark::State& state) {
  annotate_atomic_sp(state);
  std::atomic<std::shared_ptr<int>> atom(std::make_shared<int>(42));
  state.counters["lock_free"] = atom.is_lock_free() ? 1 : 0;
  for (auto _ : state) {
    auto snap = atom.load();
    benchmark::DoNotOptimize(snap);
  }
}
BENCHMARK(bm_atomic_shared_ptr_load_uncontended)->Name("std::atomic<shared_ptr<T>>::load() (uncontended)");

static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_store_uncontended(benchmark::State& state) {
  annotate_atomic_sp(state);
  std::atomic<std::shared_ptr<int>> atom;
  auto keep                   = std::make_shared<int>(7);
  state.counters["lock_free"] = atom.is_lock_free() ? 1 : 0;
  for (auto _ : state) {
    atom.store(keep);
  }
}
BENCHMARK(bm_atomic_shared_ptr_store_uncontended)->Name("std::atomic<shared_ptr<T>>::store() (uncontended)");

static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_exchange_uncontended(benchmark::State& state) {
  annotate_atomic_sp(state);
  auto keep = std::make_shared<int>(7);
  std::atomic<std::shared_ptr<int>> atom(keep);
  state.counters["lock_free"] = atom.is_lock_free() ? 1 : 0;
  for (auto _ : state) {
    auto prev = atom.exchange(keep);
    benchmark::DoNotOptimize(prev);
  }
}
BENCHMARK(bm_atomic_shared_ptr_exchange_uncontended)->Name("std::atomic<shared_ptr<T>>::exchange() (uncontended)");

static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_cas_uncontended(benchmark::State& state) {
  annotate_atomic_sp(state);
  auto keep_a = std::make_shared<int>(1);
  auto keep_b = std::make_shared<int>(2);
  std::atomic<std::shared_ptr<int>> atom(keep_a);
  state.counters["lock_free"] = atom.is_lock_free() ? 1 : 0;
  for (auto _ : state) {
    auto expected = atom.load(std::memory_order_relaxed);
    auto desired  = expected == keep_a ? keep_b : keep_a;
    (void)atom.compare_exchange_strong(expected, desired);
  }
}
BENCHMARK(bm_atomic_shared_ptr_cas_uncontended)
    ->Name("std::atomic<shared_ptr<T>>::compare_exchange_strong() (uncontended)");

struct AtomicSharedPtrSharedState {
  std::shared_ptr<int> keep_a = std::make_shared<int>(1);
  std::shared_ptr<int> keep_b = std::make_shared<int>(2);
  std::atomic<std::shared_ptr<int>> atom{keep_a};

  void reset() { atom.store(keep_a); }
};

inline void reset_shared_if_leader(benchmark::State& state, AtomicSharedPtrSharedState& shared) {
  if (state.thread_index() == 0)
    shared.reset();
}

static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_load_contended(benchmark::State& state) {
  annotate_atomic_sp(state);
  static AtomicSharedPtrSharedState shared;
  reset_shared_if_leader(state, shared);
  // Set only on thread 0: Google Benchmark sums counters across threads.
  if (state.thread_index() == 0)
    state.counters["lock_free"] = shared.atom.is_lock_free() ? 1 : 0;
  for (auto _ : state) {
    auto snap = shared.atom.load();
    benchmark::DoNotOptimize(snap);
  }
}
BENCHMARK(bm_atomic_shared_ptr_load_contended)
    ->Name("std::atomic<shared_ptr<T>>::load() (contended)")
    ->Apply(apply_contended_threads);

static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_store_contended(benchmark::State& state) {
  annotate_atomic_sp(state);
  static AtomicSharedPtrSharedState shared;
  reset_shared_if_leader(state, shared);
  if (state.thread_index() == 0)
    state.counters["lock_free"] = shared.atom.is_lock_free() ? 1 : 0;
  for (auto _ : state) {
    shared.atom.store(shared.keep_a);
  }
}
BENCHMARK(bm_atomic_shared_ptr_store_contended)
    ->Name("std::atomic<shared_ptr<T>>::store() (contended)")
    ->Apply(apply_contended_threads);

static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_cas_contended(benchmark::State& state) {
  annotate_atomic_sp(state);
  static AtomicSharedPtrSharedState shared;
  reset_shared_if_leader(state, shared);
  if (state.thread_index() == 0)
    state.counters["lock_free"] = shared.atom.is_lock_free() ? 1 : 0;
  // load() then strong CAS is a real client pattern and the lock-free
  // livelock trigger: load mutates local_count in the same DWCAS word
  // that CAS tries to replace. Do not "fix" the hang by dropping load().
  for (auto _ : state) {
    auto expected = shared.atom.load(std::memory_order_relaxed);
    auto desired  = expected == shared.keep_a ? shared.keep_b : shared.keep_a;
    (void)shared.atom.compare_exchange_strong(expected, desired);
  }
}
BENCHMARK(bm_atomic_shared_ptr_cas_contended)
    ->Name("std::atomic<shared_ptr<T>>::compare_exchange_strong() (contended)")
    ->Apply(apply_contended_threads);

// One writer, the rest load. Typical SMR-style read-mostly workload.
static TEST_ALIGN_BENCHMARK void bm_atomic_shared_ptr_readers_heavy(benchmark::State& state) {
  annotate_atomic_sp(state);
  static AtomicSharedPtrSharedState shared;
  reset_shared_if_leader(state, shared);
  if (state.thread_index() == 0)
    state.counters["lock_free"] = shared.atom.is_lock_free() ? 1 : 0;
  const bool is_writer = state.thread_index() == 0;
  for (auto _ : state) {
    if (is_writer) {
      shared.atom.store(shared.keep_b);
    } else {
      auto snap = shared.atom.load();
      benchmark::DoNotOptimize(snap);
    }
  }
}
BENCHMARK(bm_atomic_shared_ptr_readers_heavy)
    ->Name("std::atomic<shared_ptr<T>>::load() (readers-heavy, 1 writer)")
    ->Apply(apply_contended_threads);

#endif // TEST_BENCHMARK_ATOMIC_SHARED_PTR_BENCH_H
