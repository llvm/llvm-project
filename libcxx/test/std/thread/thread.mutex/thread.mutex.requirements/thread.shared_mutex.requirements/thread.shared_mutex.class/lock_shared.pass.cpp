//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// <shared_mutex>

// class shared_mutex;

// void lock_shared();

#include <shared_mutex>
#include <algorithm>
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

#include "make_test_thread.h"

int main(int, char**) {
  // Lock-shared a mutex that is not locked yet. This should succeed.
  {
    std::shared_mutex m;
    std::vector<std::thread> threads;
    for (int i = 0; i != 5; ++i) {
      threads.push_back(support::make_test_thread([&] {
        m.lock_shared();
        m.unlock_shared();
      }));
    }

    for (auto& t : threads)
      t.join();
  }

  // Lock-shared a mutex that is already exclusively locked. This should block until it is unlocked.
  {
    std::atomic<int> ready(0);
    std::shared_mutex m;
    m.lock();
    std::atomic<bool> is_locked_from_main(true);

    std::vector<std::thread> threads;
    for (int i = 0; i != 5; ++i) {
      threads.push_back(support::make_test_thread([&] {
        ++ready;
        while (ready < 5)
          /* wait until all threads have been created */;

        m.lock_shared();
        assert(!is_locked_from_main);
        m.unlock_shared();
      }));
    }

    while (ready < 5)
      /* wait until all threads have been created */;

    // We would rather signal this after we unlock, but that would create a race condition.
    // We instead signal it before we unlock, which means that it's technically possible for
    // the thread to take the lock while we're still holding it and for the test to still pass.
    is_locked_from_main = false;
    m.unlock();

    for (auto& t : threads)
      t.join();
  }

  // Lock-shared a mutex that is already lock-shared. This should succeed.
  {
    std::atomic<int> ready(0);
    std::shared_mutex m;
    m.lock_shared();

    std::vector<std::thread> threads;
    for (int i = 0; i != 5; ++i) {
      threads.push_back(support::make_test_thread([&] {
        ++ready;
        while (ready < 5)
          /* wait until all threads have been created */;

        m.lock_shared();
        m.unlock_shared();
      }));
    }

    while (ready < 5)
      /* wait until all threads have been created */;

    m.unlock_shared();

    for (auto& t : threads)
      t.join();
  }

  // Create several threads that all acquire-shared the same mutex and make sure that each
  // thread successfully acquires-shared the mutex.
  //
  // We record how many other threads were holding the mutex when it was acquired, which allows
  // us to know whether the test was somewhat effective at causing multiple threads to lock at
  // the same time.
  {
    std::shared_mutex mutex;
    std::vector<std::thread> threads;
    constexpr int n_threads           = 5;
    std::atomic<int> holders          = 0;
    int concurrent_holders[n_threads] = {};
    std::atomic<bool> ready           = false;

    for (int i = 0; i != n_threads; ++i) {
      threads.push_back(support::make_test_thread([&, i] {
        while (!ready) {
          // spin
        }

        mutex.lock_shared();
        ++holders;
        concurrent_holders[i] = holders;

        mutex.unlock_shared();
        --holders;
      }));
    }

    ready = true; // let the threads actually start shared-acquiring the mutex
    for (auto& t : threads)
      t.join();

    // We can't guarantee that we'll ever have more than 1 concurrent holder so that's what
    // we assert, however in principle we should often trigger more than 1 concurrent holder.
    int max_concurrent_holders = *std::max_element(std::begin(concurrent_holders), std::end(concurrent_holders));
    assert(max_concurrent_holders >= 1);
  }

  return 0;
}
