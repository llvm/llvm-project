//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// This is a stress test for std::atomic::wait for lost wake ups.

// <atomic>

#include <atomic>
#include <functional>
#include <thread>
#include <vector>

#include "make_test_thread.h"

constexpr int num_waiters    = 8;
constexpr int num_iterations = 10'000;

void wait(std::atomic<int>& waiter_ready, const std::atomic<int>& state) {
  for (int i = 0; i < num_iterations; ++i) {
    auto old_state = state.load(std::memory_order_acquire);
    waiter_ready.fetch_add(1, std::memory_order_acq_rel);
    state.wait(old_state, std::memory_order_acquire);
  }
}

void notify(std::atomic<int>& waiter_ready, std::atomic<int>& state) {
  for (int i = 0; i < num_iterations; ++i) {
    while (waiter_ready.load(std::memory_order_acquire) < num_waiters) {
      std::this_thread::yield();
    }
    waiter_ready.store(0, std::memory_order_release);
    state.fetch_add(1, std::memory_order_acq_rel);
    state.notify_all();
  }
}

int main(int, char**) {
  for (int run = 0; run < 20; ++run) {
    std::atomic<int> waiter_ready(0);
    std::atomic<int> state(0);
    std::vector<std::jthread> threads;
    for (int i = 0; i < 8; ++i)
      threads.push_back(support::make_test_jthread(wait, std::ref(waiter_ready), std::cref(state)));

    threads.push_back(support::make_test_jthread(notify, std::ref(waiter_ready), std::ref(state)));
  }

  return 0;
}
