//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++26

// void rcu_synchronize(rcu_domain& dom = rcu_default_domain()) noexcept;

#include <atomic>
#include <cassert>
#include <chrono>
#include <rcu>
#include <thread>
#include <vector>
#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(std::rcu_synchronize()));
static_assert(noexcept(std::rcu_synchronize(std::rcu_default_domain())));

int main(int, char**) {
  {
    constexpr size_t num_reader_threads = 4;
    constexpr size_t num_sync_threads   = 4;
    constexpr int max_reader_iteration  = 10000;
    std::array<std::atomic<int>, num_reader_threads> lock_indicies{};
    std::array<std::atomic<int>, num_reader_threads> unlock_indicies{};
    auto& dom = std::rcu_default_domain();

    std::vector<std::jthread> reader_threads;
    for (size_t i = 0; i < num_reader_threads; ++i) {
      reader_threads.push_back(support::make_test_jthread([&, i] {
        for (int count = 0; count < max_reader_iteration; ++count) {
          dom.lock();
          lock_indicies[i].store(count, std::memory_order_relaxed);
          std::this_thread::sleep_for(std::chrono::microseconds(10));

          unlock_indicies[i].store(count, std::memory_order_relaxed);
          dom.unlock();
        }
      }));
    }

    std::vector<std::jthread> sync_threads;
    for (size_t i = 0; i < num_sync_threads; ++i) {
      sync_threads.push_back(support::make_test_jthread([&](std::stop_token st) {
        while (!st.stop_requested()) {
          std::array<int, num_reader_threads> pre_sync_reader_count{};
          for (size_t j = 0; j < num_reader_threads; ++j) {
            pre_sync_reader_count[i] = lock_indicies[i].load(std::memory_order_relaxed);
          }

          std::rcu_synchronize();
          for (size_t j = 0; j < num_reader_threads; ++j) {
            assert(unlock_indicies[j].load(std::memory_order_relaxed) >= pre_sync_reader_count[j]);
          }
        }
      }));
    }

    reader_threads.clear();
    sync_threads.clear();
  }

  return 0;
}
