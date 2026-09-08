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

// void lock() noexcept;
// bool try_lock() noexcept;
// void unlock() noexcept;

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <latch>
#include <rcu>
#include <thread>
#include <utility>

#include "make_test_thread.h"
#include "test_macros.h"

static_assert(noexcept(std::declval<std::rcu_domain>().lock()));
static_assert(noexcept(std::declval<std::rcu_domain>().unlock()));
static_assert(noexcept(std::declval<std::rcu_domain>().try_lock()));

int main(int, char**) {
  {
    // lock opens rcu protected region
    // unlock closes it
    std::atomic_bool flag = false;
    auto& dom             = std::rcu_default_domain();
    dom.lock();

    auto t = support::make_test_jthread([&] {
      std::rcu_synchronize();
      flag = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    assert(flag.load() == false);
    dom.unlock(); // should unblock rcu_synchronize
  }

  {
    // lock/unlock can be nested

    std::atomic_bool flag = false;
    auto& dom             = std::rcu_default_domain();
    dom.lock();
    dom.lock();
    dom.lock();

    auto t = support::make_test_jthread([&] {
      std::rcu_synchronize();
      flag = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    assert(flag.load() == false);

    dom.unlock();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    assert(flag.load() == false);

    dom.unlock();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    assert(flag.load() == false);

    dom.unlock();
  }

  {
    // multi-threaded lock/unlock
    constexpr std::size_t num_threads = 4;
    std::atomic_bool flag             = false;
    std::array<std::atomic_bool, 4> unlock_signal{};
    std::latch lock_latch(num_threads);

    std::vector<std::jthread> threads;
    threads.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
      threads.push_back(support::make_test_jthread([&, i] {
        auto& dom = std::rcu_default_domain();
        dom.lock();
        lock_latch.arrive_and_wait();
        unlock_signal[i].wait(false);
        dom.unlock();
      }));
    }

    auto sync_thread = support::make_test_jthread([&] {
      lock_latch.wait();
      std::rcu_synchronize();
      flag = true;
    });

    lock_latch.wait();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(flag.load() == false);

    for (std::size_t i = 0; i < num_threads - 1; ++i) {
      unlock_signal[i].store(true);
      unlock_signal[i].notify_all();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      assert(flag.load() == false);
    }

    unlock_signal[3].store(true);
    unlock_signal[3].notify_all();
  }
  {
    // try_lock
    std::atomic_bool flag               = false;
    auto& dom                           = std::rcu_default_domain();
    std::same_as<bool> decltype(auto) r = dom.try_lock();
    assert(r);

    auto t = support::make_test_jthread([&] {
      std::rcu_synchronize();
      flag = true;
    });

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    // flag should not be set because rcu_synchronize should block after lock
    assert(flag.load() == false);
    dom.unlock(); // should unblock rcu_synchronize
  }

  return 0;
}
