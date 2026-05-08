//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

#include <mutex>
#include <thread>
#include <vector>
#include <atomic>
#include <benchmark/benchmark.h>

// Steady state: flag already _Complete, never enters __call_once.
// Measures the inline header fast-path only.
static void BM_call_once_steady(benchmark::State& state) {
  std::once_flag f;
  std::call_once(f, [] {});
  for (auto _ : state) {
    std::call_once(f, [] {});
    benchmark::ClobberMemory();
  }
}
BENCHMARK(BM_call_once_steady);

// Steady state under contention: N threads hammer an already-complete flag.
// Measures whether the acquire-load scales across cores.
static void BM_call_once_steady_contended(benchmark::State& state) {
  std::once_flag f;
  std::call_once(f, [] {});

  for (auto _ : state)
    std::call_once(f, [] {});
}
BENCHMARK(BM_call_once_steady_contended)->Threads(2)->Threads(4)->Threads(8)->Threads(16);

// Cold path: fresh flag each iteration, single thread.
// Measures one full trip through __call_once (CAS or mutex path).
static void BM_call_once_cold(benchmark::State& state) {
  for (auto _ : state) {
    std::once_flag f;
    std::call_once(f, [] {});
    benchmark::DoNotOptimize(f);
  }
}
BENCHMARK(BM_call_once_cold);

// Contended: N threads race on a fresh flag.
// One wins and runs func, the rest wait then return.
// Measures the full contended path including wait/wake.
static void BM_call_once_contended(benchmark::State& state) {
  const int nthreads = state.range(0);

  for (auto _ : state) {
    state.PauseTiming();

    std::once_flag flag;
    std::atomic<bool> go{false};
    std::atomic<int> ready{0};
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    for (int i = 0; i < nthreads; ++i) {
      threads.emplace_back([&] {
        ready.fetch_add(1, std::memory_order_relaxed);
        while (!go.load(std::memory_order_acquire)) {
        }
        std::call_once(flag, [] {});
      });
    }

    // Wait for all threads to be ready
    while (ready.load(std::memory_order_relaxed) < nthreads) {
    }

    state.ResumeTiming();
    go.store(true, std::memory_order_release);

    for (auto& t : threads)
      t.join();
  }

  state.SetItemsProcessed(state.iterations() * nthreads);
}
BENCHMARK(BM_call_once_contended)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

// Contended with slow init: func takes real time, waiters must block.
// Shows cost of wait/wake mechanism under realistic conditions.
static void BM_call_once_slow_init(benchmark::State& state) {
  const int nthreads = state.range(0);

  for (auto _ : state) {
    state.PauseTiming();

    std::once_flag flag;
    std::atomic<bool> go{false};
    std::atomic<int> ready{0};
    int shared_data = 0;
    std::vector<std::thread> threads;
    threads.reserve(nthreads);

    for (int i = 0; i < nthreads; ++i) {
      threads.emplace_back([&] {
        ready.fetch_add(1, std::memory_order_relaxed);
        while (!go.load(std::memory_order_acquire)) {
        }
        std::call_once(flag, [&] { benchmark::DoNotOptimize(shared_data = 42); });
        benchmark::DoNotOptimize(shared_data);
      });
    }

    while (ready.load(std::memory_order_relaxed) < nthreads) {
    }

    state.ResumeTiming();
    go.store(true, std::memory_order_release);

    for (auto& t : threads)
      t.join();
  }
}
BENCHMARK(BM_call_once_slow_init)->Arg(2)->Arg(4)->Arg(8)->Arg(16);

// Throughput: many fresh flags in sequence, single thread.
// Measures raw cold-path throughput without thread overhead.
static void BM_call_once_throughput(benchmark::State& state) {
  for (auto _ : state) {
    for (int i = 0; i < 1000; ++i) {
      std::once_flag f;
      std::call_once(f, [] {});
      benchmark::DoNotOptimize(f);
    }
  }
  state.SetItemsProcessed(state.iterations() * 1000);
}
BENCHMARK(BM_call_once_throughput);

BENCHMARK_MAIN();
