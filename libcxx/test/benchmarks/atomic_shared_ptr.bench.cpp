//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: std-at-least-c++20

#include <atomic>
#include <memory>

#include "benchmark/benchmark.h"

static void BM_AtomicSharedPtrLoadUncontended(benchmark::State& st) {
  std::atomic<std::shared_ptr<int>> atom(std::make_shared<int>(42));
  while (st.KeepRunning()) {
    auto snap = atom.load();
    benchmark::DoNotOptimize(snap);
  }
}
BENCHMARK(BM_AtomicSharedPtrLoadUncontended);

static void BM_AtomicSharedPtrStoreUncontended(benchmark::State& st) {
  std::atomic<std::shared_ptr<int>> atom;
  auto keep = std::make_shared<int>(7);
  while (st.KeepRunning()) {
    atom.store(keep);
  }
}
BENCHMARK(BM_AtomicSharedPtrStoreUncontended);

BENCHMARK_MAIN();
