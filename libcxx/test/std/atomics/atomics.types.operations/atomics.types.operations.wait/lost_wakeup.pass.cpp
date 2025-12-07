//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: !stdlib=system 

// This is a stress test for std::atomic::wait for lost wake ups.

// <atomic>

#include <atomic>
#include <chrono>
#include <functional>
#include <thread>
#include <vector>
#include <iostream>
#include <iomanip>

#include "make_test_thread.h"

constexpr int num_waiters    = 8;
constexpr int num_iterations = 10'000;

int main(int, char**) {
  auto start = std::chrono::high_resolution_clock::now();
  for (int run = 0; run < 20; ++run) {
    std::cerr << "run " << run << std::endl;
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
        if (i % 1000 == 0)
          std::cerr << std::fixed << std::setprecision(2) << std::left
                    << std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - start) << " run "
                    << run << "  notify iteration " << i << std::endl;

        while (waiter_ready.load() < num_waiters) {
          std::this_thread::yield();
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

  return 1;
}
