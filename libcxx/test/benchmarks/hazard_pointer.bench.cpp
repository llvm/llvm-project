//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++26
// UNSUPPORTED: no-threads
// XFAIL: availability-hazard_pointer-missing

#include <hazard_pointer>
#include <atomic>
#include <cstddef>
#include <memory>
#include <vector>

#include "benchmark/benchmark.h"

namespace {

struct Node : std::hazard_pointer_obj_base<Node> {
  std::size_t value = 42;
};

// make + destroy a nonempty hazard_pointer per iteration (thread-cache hit in the steady state).
void BM_make_destroy(benchmark::State& state) {
  for (auto _ : state) {
    std::hazard_pointer h = std::make_hazard_pointer();
    benchmark::DoNotOptimize(h);
  }
}

// protect() per iteration with one hazard_pointer made once; the protection is left in place.
void BM_protect_combined(benchmark::State& state) {
  auto own = std::make_unique<Node>();
  std::atomic<Node*> src{own.get()};
  benchmark::DoNotOptimize(src);
  std::hazard_pointer h = std::make_hazard_pointer();
  std::size_t sum       = 0;
  for (auto _ : state) {
    Node* p = h.protect(src);
    benchmark::DoNotOptimize(p->value);
    sum += p->value;
  }
  h.reset_protection();
  benchmark::DoNotOptimize(sum);
}

// protect() + reset_protection() per iteration.
void BM_protect_separate(benchmark::State& state) {
  auto own = std::make_unique<Node>();
  std::atomic<Node*> src{own.get()};
  benchmark::DoNotOptimize(src);
  std::hazard_pointer h = std::make_hazard_pointer();
  std::size_t sum       = 0;
  for (auto _ : state) {
    Node* p = h.protect(src);
    benchmark::DoNotOptimize(p->value);
    sum += p->value;
    h.reset_protection();
  }
  benchmark::DoNotOptimize(sum);
}

// make + protect + destroy per iteration.
void BM_make_protect(benchmark::State& state) {
  auto own = std::make_unique<Node>();
  std::atomic<Node*> src{own.get()};
  benchmark::DoNotOptimize(src);
  std::size_t sum = 0;
  for (auto _ : state) {
    std::hazard_pointer h = std::make_hazard_pointer();
    Node* p               = h.protect(src);
    benchmark::DoNotOptimize(p->value);
    sum += p->value;
  }
  benchmark::DoNotOptimize(sum);
}

// retire() of a pre-allocated object per iteration (allocation happens in untimed batches); reclamation
// runs inline every ~1000 retirements and is part of the measurement, as it is in real use.
void BM_retire(benchmark::State& state) {
  constexpr std::size_t kBatch = std::size_t{1} << 16;
  std::vector<std::unique_ptr<Node>> objs;
  std::size_t next = kBatch;
  for (auto _ : state) {
    if (next == kBatch) {
      state.PauseTiming();
      objs.clear();
      objs.reserve(kBatch);
      for (std::size_t i = 0; i < kBatch; ++i)
        objs.push_back(std::make_unique<Node>());
      next = 0;
      state.ResumeTiming();
    }
    Node* n = objs[next++].release();
    n->retire();
  }
}

// Multi-threaded: every thread makes and destroys hazard pointers. Deliberately the same body as
// BM_make_destroy, so the two are directly comparable: the difference is the cost of contention.
void BM_mt_make_destroy(benchmark::State& state) {
  for (auto _ : state) {
    std::hazard_pointer h = std::make_hazard_pointer();
    benchmark::DoNotOptimize(h);
  }
}

// Multi-threaded: every thread allocates and retires objects (contended retired lists + reclamation).
void BM_mt_retire(benchmark::State& state) {
  for (auto _ : state) {
    Node* n = new Node;
    n->retire();
  }
}

} // namespace

BENCHMARK(BM_make_destroy)->Name("std::make_hazard_pointer() (make + destroy)");
BENCHMARK(BM_protect_combined)->Name("std::hazard_pointer::protect(const atomic<T*>&) (protection kept)");
BENCHMARK(BM_protect_separate)->Name("std::hazard_pointer::protect(const atomic<T*>&) (+ reset_protection())");
BENCHMARK(BM_make_protect)->Name("std::make_hazard_pointer() (+ protect(), destroy)");
BENCHMARK(BM_retire)->Name("std::hazard_pointer_obj_base<T>::retire()");
BENCHMARK(BM_mt_make_destroy)
    ->Name("std::make_hazard_pointer() (make + destroy, contended)")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);
BENCHMARK(BM_mt_retire)
    ->Name("std::hazard_pointer_obj_base<T>::retire() (contended)")
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->Threads(8);

BENCHMARK_MAIN();
