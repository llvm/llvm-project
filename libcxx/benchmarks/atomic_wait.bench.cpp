//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <atomic>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <thread>

#include "benchmark/benchmark.h"
#include "make_test_thread.h"

using namespace std::chrono_literals;

struct HighPrioTask {
  sched_param param;
  pthread_attr_t attr_t;
  pthread_t thread;
  std::atomic_bool stopped{false};

  HighPrioTask(const HighPrioTask&) = delete;

  HighPrioTask() {
    pthread_attr_init(&attr_t);
    pthread_attr_setschedpolicy(&attr_t, SCHED_FIFO);
    param.sched_priority = sched_get_priority_max(SCHED_FIFO);
    pthread_attr_setschedparam(&attr_t, &param);
    pthread_attr_setinheritsched(&attr_t, PTHREAD_EXPLICIT_SCHED);

    auto thread_fun = [](void* arg) -> void* {
      auto* stop = reinterpret_cast<std::atomic_bool*>(arg);
      while (!stop->load(std::memory_order_relaxed)) {
        // spin
      }
      return nullptr;
    };

    if (pthread_create(&thread, &attr_t, thread_fun, &stopped) != 0) {
      throw std::runtime_error("failed to create thread");
    }
  }

  ~HighPrioTask() {
    stopped = true;
    pthread_attr_destroy(&attr_t);
    pthread_join(thread, nullptr);
  }
};


template <std::size_t N>
struct NumHighPrioTasks {
  static constexpr auto value = N;
};


struct KeepNotifying {
  template <class Atomic>
  static void notify(Atomic& a, std::stop_token st) {
    while (!st.stop_requested()) {
      a.fetch_add(1, std::memory_order_relaxed);
      a.notify_all();
    }
  }
};

template <std::size_t N>
struct NotifyEveryNus {
  template <class Atomic>
  static void notify(Atomic& a, std::stop_token st) {
    while (!st.stop_requested()) {
      auto start = std::chrono::system_clock::now();
      a.fetch_add(1, std::memory_order_relaxed);
      a.notify_all();
      while (std::chrono::system_clock::now() - start < std::chrono::microseconds{N}) {
      }
    }
  }
};

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

BENCHMARK(BM_1_atomic_1_waiter_1_notifier<KeepNotifying, NumHighPrioTasks<0>>)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<50>, NumHighPrioTasks<0>>)->RangeMultiplier(2)->Range(1 << 10, 1 << 16);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<100>, NumHighPrioTasks<0>>)->RangeMultiplier(2)->Range(1 << 10, 1 << 16);

BENCHMARK(BM_1_atomic_1_waiter_1_notifier<KeepNotifying, NumHighPrioTasks<4>>)->RangeMultiplier(2)->Range(1 << 10, 1 << 24);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<50>, NumHighPrioTasks<4>>)->RangeMultiplier(2)->Range(1 << 10, 1 << 16);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<100>, NumHighPrioTasks<4>>)->RangeMultiplier(2)->Range(1 << 10, 1 << 16);

BENCHMARK(BM_1_atomic_1_waiter_1_notifier<KeepNotifying, NumHighPrioTasks<7>>)->RangeMultiplier(2)->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<50>, NumHighPrioTasks<7>>)->RangeMultiplier(2)->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_1_waiter_1_notifier<NotifyEveryNus<100>, NumHighPrioTasks<7>>)->RangeMultiplier(2)->Range(1 << 4, 1 << 8);


template <std::size_t N>
struct NumWaitingThreads {
  static constexpr auto value = N;
};

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
    ->Range(1 << 10, 1 << 20);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<15>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 20);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<15>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 16);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<3>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<7>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<15>, NumHighPrioTasks<0>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14);


BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 18);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 18);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<15>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 18);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 14);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 14);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<15>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 10, 1 << 14);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<3>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<7>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<15>, NumHighPrioTasks<4>>)
    ->RangeMultiplier(2)
    ->Range(1 << 8, 1 << 14);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<3>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<7>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<KeepNotifying, NumWaitingThreads<15>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<3>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<7>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<50>, NumWaitingThreads<15>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);

BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<3>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<7>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);
BENCHMARK(BM_1_atomic_multi_waiter_1_notifier<NotifyEveryNus<100>, NumWaitingThreads<15>, NumHighPrioTasks<7>>)
    ->RangeMultiplier(2)
    ->Range(1 << 4, 1 << 8);


template <std::size_t N>
struct NumberOfAtomics {
  static constexpr auto value = N;
};

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
     ->Range(1 << 10, 1 << 20);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<3>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 20);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<5>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 20);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<7>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 20);

 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<2>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<3>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<5>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<7>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);

 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<2>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 8, 1 << 14);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<3>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 8, 1 << 14);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<5>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 8, 1 << 14);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<7>, NumHighPrioTasks<0>>)
     ->RangeMultiplier(2)
     ->Range(1 << 8, 1 << 14);

 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<2>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 20);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<3>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 20);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<5>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 20);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<KeepNotifying, NumberOfAtomics<7>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 20);

 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<2>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<3>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<5>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<50>, NumberOfAtomics<7>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 10, 1 << 16);


 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<2>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 8, 1 << 14);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<3>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 8, 1 << 14);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<5>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 6, 1 << 10);
 BENCHMARK(BM_N_atomics_N_waiter_N_notifier<NotifyEveryNus<100>, NumberOfAtomics<7>, NumHighPrioTasks<4>>)
     ->RangeMultiplier(2)
     ->Range(1 << 4, 1 << 8);

BENCHMARK_MAIN();
