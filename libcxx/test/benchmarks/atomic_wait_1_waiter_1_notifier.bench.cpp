//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include "atomic_wait_helper.h"

#include <atomic>
#include <array>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <stop_token>
#include <thread>

#include "benchmark/benchmark.h"
#include "make_test_thread.h"

using namespace std::chrono_literals;

template <class NotifyPolicy, class NumPrioTasks>
void BM_1_atomic_1_waiter_1_notifier(benchmark::State& state) {
  [[maybe_unused]] std::array<HighPrioTask, NumPrioTasks::value> tasks{};
  std::atomic<std::uint64_t> a;
  auto thread_func = [&](std::stop_token st) { NotifyPolicy::notify(a, st); };

  std::uint64_t total_loop_test_param = state.range(0);

  auto thread = support::make_test_jthread(thread_func);

  for (auto _ : state) {
    for (std::uint64_t i = 0; i < total_loop_test_param; ++i) {
      auto old = a.load(std::memory_order_relaxed);
      a.wait(old);
    }
  }
}

BENCHMARK(BM_1_atomic_1_waiter_1_notifier<KeepNotifying, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1 << 18);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<50>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<100>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);

BENCHMARK(BM_1_atomic_1_waiter_1_notifier<KeepNotifying, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 16, 1 << 18);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<50>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<100>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);

BENCHMARK(BM_1_atomic_1_waiter_1_notifier<KeepNotifying, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<50>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 5);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<100>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 5);

BENCHMARK_MAIN();
