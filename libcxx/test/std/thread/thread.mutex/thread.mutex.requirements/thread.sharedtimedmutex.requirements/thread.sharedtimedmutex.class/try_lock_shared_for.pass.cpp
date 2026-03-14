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

// template <class Rep, class Period>
//     bool try_lock_shared_for(const chrono::duration<Rep, Period>& rel_time);

#include <shared_mutex>
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>
#include <vector>

#include "make_test_thread.h"

template <class Function>
std::chrono::microseconds measure(Function f) {
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  f();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

int main(int, char**) {
  // Try to lock-shared a mutex that is not locked yet. This should succeed immediately.
  {
    std::shared_timed_mutex m;
    std::vector<std::thread> threads;
    for (int i = 0; i != 5; ++i) {
      threads.push_back(support::make_test_thread([&] {
        bool succeeded = m.try_lock_shared_for(std::chrono::milliseconds(1));
        assert(succeeded);
        m.unlock_shared();
      }));
    }

    for (auto& t : threads)
      t.join();
  }

  // Try to lock-shared an already-locked mutex for a long enough amount of time and succeed.
  // This is technically flaky, but we use such long durations that it should pass even
  // in slow or contended environments.
  {
    std::chrono::milliseconds const wait_time(500);
    std::chrono::milliseconds const tolerance = wait_time * 3;
    std::atomic<int> ready(0);

    std::shared_timed_mutex m;
    m.lock();

    std::vector<std::thread> threads;
    for (int i = 0; i != 5; ++i) {
      threads.push_back(support::make_test_thread([&] {
        ++ready;
        while (ready < 5)
          /* spin until all threads are created */;

        auto elapsed = measure([&] {
          bool succeeded = m.try_lock_shared_for(wait_time);
          assert(succeeded);
          m.unlock_shared();
        });

        // Ensure we didn't wait significantly longer than our timeout. This is technically
        // flaky and non-conforming because an implementation is free to block for arbitrarily
        // long, but any decent quality implementation should pass this test.
        assert(elapsed - wait_time < tolerance);
      }));
    }

    // Wait for all the threads to be ready to take the lock before we unlock it from here, otherwise
    // there's a high chance that we're not testing the "locking an already locked" mutex use case.
    // There is still technically a race condition here.
    while (ready < 5)
      /* spin */;
    std::this_thread::sleep_for(wait_time / 5);

    m.unlock(); // this should allow the threads to lock-shared 'm'

    for (auto& t : threads)
      t.join();
  }

  // Try to lock-shared an already-locked mutex for a short amount of time and fail.
  // Again, this is technically flaky but we use such long durations that it should work.
  {
    std::chrono::milliseconds const wait_time(10);
    std::chrono::milliseconds const tolerance(750); // in case the thread we spawned goes to sleep or something

    std::shared_timed_mutex m;
    m.lock();

    std::vector<std::thread> threads;
    for (int i = 0; i != 5; ++i) {
      threads.push_back(support::make_test_thread([&] {
        auto elapsed = measure([&] {
          bool succeeded = m.try_lock_shared_for(wait_time);
          assert(!succeeded);
        });

        // Ensure we failed within some bounded time.
        assert(elapsed - wait_time < tolerance);
      }));
    }

    for (auto& t : threads)
      t.join();

    m.unlock();
  }

  return 0;
}
