//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11

// <shared_mutex>

// class shared_timed_mutex;

// void lock();

#include <shared_mutex>
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>

#include "make_test_thread.h"

int main(int, char**) {
  // Exclusive-lock a mutex that is not locked yet. This should succeed.
  {
    std::shared_timed_mutex m;
    m.lock();
    m.unlock();
  }

  // Exclusive-lock a mutex that is already locked exclusively. This should block until it is unlocked.
  {
    std::atomic<bool> ready(false);
    std::shared_timed_mutex m;
    m.lock();
    std::atomic<bool> is_locked_from_main(true);

    std::thread t = support::make_test_thread([&] {
      ready = true;
      m.lock();
      assert(!is_locked_from_main);
      m.unlock();
    });

    while (!ready)
      /* spin */;

    // We would rather signal this after we unlock, but that would create a race condition.
    // We instead signal it before we unlock, which means that it's technically possible for the thread
    // to take the lock while we're still holding it and for the test to still pass.
    is_locked_from_main = false;
    m.unlock();

    t.join();
  }

  // Exclusive-lock a mutex that is already share-locked. This should block until it is unlocked.
  {
    std::atomic<bool> ready(false);
    std::shared_timed_mutex m;
    m.lock_shared();
    std::atomic<bool> is_locked_from_main(true);

    std::thread t = support::make_test_thread([&] {
      ready = true;
      m.lock();
      assert(!is_locked_from_main);
      m.unlock();
    });

    while (!ready)
      /* spin */;

    // We would rather signal this after we unlock, but that would create a race condition.
    // We instead signal it before we unlock, which means that it's technically possible for
    // the thread to take the lock while we're still holding it and for the test to still pass.
    is_locked_from_main = false;
    m.unlock_shared();

    t.join();
  }

  // Make sure that at most one thread can acquire the mutex concurrently.
  {
    std::atomic<int> counter(0);
    std::shared_timed_mutex mutex;

    std::vector<std::thread> threads;
    for (int i = 0; i != 10; ++i) {
      threads.push_back(support::make_test_thread([&] {
        mutex.lock();
        counter++;
        assert(counter == 1);
        counter--;
        mutex.unlock();
      }));
    }

    for (auto& t : threads)
      t.join();
  }

  return 0;
}
