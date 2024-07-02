//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads, c++03

// <condition_variable>

// class condition_variable;

// template <class Rep, class Period>
//     cv_status
//     wait_for(unique_lock<mutex>& lock,
//              const chrono::duration<Rep, Period>& rel_time);

#include <condition_variable>
#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

template <class Function>
std::chrono::microseconds measure(Function f) {
  std::chrono::high_resolution_clock::time_point start = std::chrono::high_resolution_clock::now();
  f();
  std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration_cast<std::chrono::microseconds>(end - start);
}

int main(int, char**) {
  // Test unblocking via a call to notify_one() in another thread.
  //
  // To test this, we set a very long timeout in wait_for() and we wait
  // again in case we get awoken spuriously. Note that it can actually
  // happen that we get awoken spuriously and fail to recognize it
  // (making this test useless), but the likelihood should be small.
  {
    std::atomic<bool> ready(false);
    std::atomic<bool> likely_spurious(true);
    auto timeout = std::chrono::seconds(3600);
    std::condition_variable cv;
    std::mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      std::unique_lock<std::mutex> lock(mutex);
      auto elapsed = measure([&] {
        ready = true;
        do {
          std::cv_status result = cv.wait_for(lock, timeout);
          assert(result == std::cv_status::no_timeout);
        } while (likely_spurious);
      });

      // This can technically fail if we have many spurious awakenings, but in practice the
      // tolerance is so high that it shouldn't be a problem.
      assert(elapsed < timeout);
    });

    std::thread t2 = support::make_test_thread([&] {
      while (!ready) {
        // spin
      }

      // Acquire the same mutex as t1. This blocks the condition variable inside its wait call
      // so we can notify it while it is waiting.
      std::unique_lock<std::mutex> lock(mutex);
      cv.notify_one();
      likely_spurious = false;
      lock.unlock();
    });

    t2.join();
    t1.join();
  }

  // Test unblocking via a timeout.
  //
  // To test this, we create a thread that waits on a condition variable
  // with a certain timeout, and we never awaken it. To guard against
  // spurious wakeups, we wait again whenever we are awoken for a reason
  // other than a timeout.
  {
    auto timeout = std::chrono::milliseconds(250);
    std::condition_variable cv;
    std::mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      std::unique_lock<std::mutex> lock(mutex);
      std::cv_status result;
      do {
        auto elapsed = measure([&] { result = cv.wait_for(lock, timeout); });
        if (result == std::cv_status::timeout)
          assert(elapsed >= timeout);
      } while (result != std::cv_status::timeout);
    });

    t1.join();
  }

  return 0;
}
