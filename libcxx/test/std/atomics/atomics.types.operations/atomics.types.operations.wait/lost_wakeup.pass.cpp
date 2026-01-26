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

int main(int, char**) {
  for (int run = 0; run < 20; ++run) {
    std::atomic<int> waiter_ready(0);
    std::atomic<int> state(0);

    auto wait = [&]() {
      for (int i = 0; i < num_iterations; ++i) {
        auto old_state = state.load();
        waiter_ready.fetch_add(1);
        state.wait(old_state);
      }
    };

    auto notify = [&] {
      for (int i = 0; i < num_iterations; ++i) {
        while (waiter_ready.load() < num_waiters) {
        }
        waiter_ready.store(0);
        state.fetch_add(1);
        state.notify_all();
      }
    };

    std::vector<std::jthread> threads;
    for (int i = 0; i < num_waiters; ++i)
      threads.push_back(support::make_test_jthread(wait));

    threads.push_back(support::make_test_jthread(notify));
  }

  return 0;
}
