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
#include <pthread.h>
#include <sched.h>
#include <thread>
#include <chrono>
#include <array>

#include "benchmark/benchmark.h"
#include "make_test_thread.h"

using namespace std::chrono_literals;

template <class NotifyPolicy, class NumberOfAtomics, class NumPrioTasks>
void BM_N_atomics_N_waiter_N_notifier(benchmark::State& state) {
  [[maybe_unused]] std::array<HighPrioTask, NumPrioTasks::value> tasks{};
  const std::uint64_t total_loop_test_param = state.range(0);
  constexpr std::uint64_t num_atomics       = NumberOfAtomics::value;
  std::vector<std::atomic<std::uint64_t>> atomics(num_atomics);

  auto notify_func = [&](std::stop_token st, size_t idx) {
    while (!st.stop_requested()) {
      NotifyPolicy::notify(atomics[idx], st);
    }
  };

  std::atomic<std::uint64_t> start_flag = 0;
  std::atomic<std::uint64_t> done_count = 0;

  auto wait_func = [&, total_loop_test_param](std::stop_token st, size_t idx) {
    auto old_start = 0;
    while (!st.stop_requested()) {
      start_flag.wait(old_start);
      old_start = start_flag.load();
      for (std::uint64_t i = 0; i < total_loop_test_param; ++i) {
        auto old = atomics[idx].load(std::memory_order_relaxed);
        atomics[idx].wait(old);
      }
      done_count.fetch_add(1);
    }
  };

  std::vector<std::jthread> notify_threads;
  notify_threads.reserve(num_atomics);

  std::vector<std::jthread> wait_threads;
  wait_threads.reserve(num_atomics);

  for (size_t i = 0; i < num_atomics; ++i) {
    notify_threads.emplace_back(support::make_test_jthread(notify_func, i));
  }

  for (size_t i = 0; i < num_atomics; ++i) {
    wait_threads.emplace_back(support::make_test_jthread(wait_func, i));
  }

  for (auto _ : state) {
    done_count = 0;
    start_flag.fetch_add(1);
    start_flag.notify_all();
    while (done_count < num_atomics) {
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

BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<2>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 12, 1 << 14);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<5>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);

BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<2>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 12);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<5>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);

BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<2>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 10);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<5>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 9);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);

BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<2>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 9);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 9);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<5>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 6);

BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<2>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 9);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 7, 1 << 9);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<5>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 7);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 5);

BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<2>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 6, 1 << 8);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<5>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 5, 1 << 7);
BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 3, 1 << 5);

BENCHMARK_MAIN();
