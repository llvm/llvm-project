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
#include <cstdint>
#include <numeric>
#include <stop_token>
#include <thread>
#include <chrono>
#include <array>

#include "benchmark/benchmark.h"
#include "make_test_thread.h"

using namespace std::chrono_literals;

template <class NotifyPolicy, class NumWaitingThreads, class NumPrioTasks>
void BM_1_atomic_multi_waiter_1_notifier(benchmark::State& state) {
  [[maybe_unused]] std::array<HighPrioTask, NumPrioTasks::value> tasks{};

  std::atomic<std::uint64_t> a;
  auto notify_func = [&](std::stop_token st) { NotifyPolicy::notify(a, st); };

  std::uint64_t total_loop_test_param = state.range(0);
  constexpr auto num_waiting_threads  = NumWaitingThreads::value;
  std::vector<std::jthread> wait_threads;
  wait_threads.reserve(num_waiting_threads);

  auto notify_thread = support::make_test_jthread(notify_func);

  std::atomic<std::uint64_t> start_flag = 0;
  std::atomic<std::uint64_t> done_count = 0;
  auto wait_func                        = [&a, &start_flag, &done_count, total_loop_test_param](std::stop_token st) {
    auto old_start = 0;
    while (!st.stop_requested()) {
      start_flag.wait(old_start);
      old_start = start_flag.load();
      for (std::uint64_t i = 0; i < total_loop_test_param; ++i) {
        auto old = a.load(std::memory_order_relaxed);
        a.wait(old);
      }
      done_count.fetch_add(1);
    }
  };

  for (size_t i = 0; i < num_waiting_threads; ++i) {
    wait_threads.emplace_back(support::make_test_jthread(wait_func));
  }

  for (auto _ : state) {
    done_count = 0;
    start_flag.fetch_add(1);
    start_flag.notify_all();
    while (done_count < num_waiting_threads) {
      std::this_thread::yield();
    }
  }
  for (auto& t : wait_threads) {
    t.request_stop();
  }
  start_flag.fetch_add(1);
  start_flag.notify_all();
  for (auto& t : wait_threads) {
    t.join();
  }
}

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 14, 1 << 16);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 14);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<15>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<15>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<15>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<15>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<15>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<15>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<3>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<7>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 5);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<15>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 2, 1 << 4);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<3>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 5);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<7>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 2, 1 << 4);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<15>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 3);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<3>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 5);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<7>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 2, 1 << 4);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<15>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 1, 1 << 3);

BENCHMARK_MAIN();
