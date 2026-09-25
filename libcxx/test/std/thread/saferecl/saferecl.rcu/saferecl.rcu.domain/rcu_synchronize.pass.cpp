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
    // rcu_synchronize();

    std::atomic_bool sync_returned = false;
    std::atomic_bool unlock_called = false;
    auto& dom                      = std::rcu_default_domain();
    dom.lock();

    auto t = support::make_test_jthread([&] {
      std::rcu_synchronize();
      assert(unlock_called.load(std::memory_order_relaxed));
      sync_returned = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    assert(!sync_returned.load(std::memory_order_relaxed));

    unlock_called.store(true, std::memory_order_relaxed);
    dom.unlock(); // should unblock rcu_synchronize
  }

  {
    // rcu_synchronize(dom);

    std::atomic_bool sync_returned = false;
    std::atomic_bool unlock_called = false;
    auto& dom                      = std::rcu_default_domain();
    dom.lock();

    auto t = support::make_test_jthread([&] {
      std::rcu_synchronize(dom);
      assert(unlock_called.load(std::memory_order_relaxed));
      sync_returned = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    assert(!sync_returned.load(std::memory_order_relaxed));

    unlock_called.store(true, std::memory_order_relaxed);
    dom.unlock(); // should unblock rcu_synchronize
  }
  {
    // multithreaded synchronize
    constexpr size_t num_sync_threads = 4;
    std::array<std::atomic_bool, num_sync_threads> sync_returned{};
    std::atomic_bool unlock_called = false;
    auto& dom                      = std::rcu_default_domain();
    dom.lock();

    std::vector<std::jthread> sync_threads;
    for (size_t i = 0; i < num_sync_threads; ++i) {
      sync_threads.push_back(support::make_test_jthread([&, i] {
        std::rcu_synchronize();
        assert(unlock_called.load(std::memory_order_relaxed));
        sync_returned[i].store(true, std::memory_order_relaxed);
      }));
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    for (auto&& sync_ret : sync_returned) {
      assert(!sync_ret.load(std::memory_order_relaxed));
    }

    unlock_called.store(true, std::memory_order_relaxed);
    dom.unlock(); // should unblock rcu_synchronize
  }

  return 0;
}
