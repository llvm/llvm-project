//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads, c++03

// <condition_variable>

// class condition_variable_any;

// template <class Lock, class Clock, class Duration>
//   cv_status
//   wait_until(Lock& lock, const chrono::time_point<Clock, Duration>& abs_time);

#include <condition_variable>
#include <atomic>
#include <cassert>
#include <chrono>
#include <mutex>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

struct TestClock {
  typedef std::chrono::milliseconds duration;
  typedef duration::rep rep;
  typedef duration::period period;
  typedef std::chrono::time_point<TestClock> time_point;
  static const bool is_steady = true;

  static time_point now() {
    using namespace std::chrono;
    return time_point(duration_cast<duration>(steady_clock::now().time_since_epoch()));
  }
};

template <class Mutex>
struct MyLock : std::unique_lock<Mutex> {
  using std::unique_lock<Mutex>::unique_lock;
};

template <class Lock, class Clock>
void test() {
  using Mutex = typename Lock::mutex_type;
  // Test unblocking via a call to notify_one() in another thread.
  //
  // To test this, we set a very long timeout in wait_until() and we wait
  // again in case we get awoken spuriously. Note that it can actually
  // happen that we get awoken spuriously and fail to recognize it
  // (making this test useless), but the likelihood should be small.
  {
    std::atomic<bool> ready(false);
    std::atomic<bool> likely_spurious(true);
    auto timeout = Clock::now() + std::chrono::seconds(3600);
    std::condition_variable_any cv;
    Mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      Lock lock(mutex);
      ready = true;
      do {
        std::cv_status result = cv.wait_until(lock, timeout);
        assert(result == std::cv_status::no_timeout);
      } while (likely_spurious);

      // This can technically fail if we have many spurious awakenings, but in practice the
      // tolerance is so high that it shouldn't be a problem.
      assert(Clock::now() < timeout);
    });

    std::thread t2 = support::make_test_thread([&] {
      while (!ready) {
        // spin
      }

      // Acquire the same mutex as t1. This blocks the condition variable inside its wait call
      // so we can notify it while it is waiting.
      Lock lock(mutex);
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
    auto timeout = Clock::now() + std::chrono::milliseconds(250);
    std::condition_variable_any cv;
    Mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      Lock lock(mutex);
      std::cv_status result;
      do {
        result = cv.wait_until(lock, timeout);
        if (result == std::cv_status::timeout)
          assert(Clock::now() >= timeout);
      } while (result != std::cv_status::timeout);
    });

    t1.join();
  }
}

int main(int, char**) {
  test<std::unique_lock<std::mutex>, TestClock>();
  test<std::unique_lock<std::mutex>, std::chrono::steady_clock>();

  test<std::unique_lock<std::timed_mutex>, TestClock>();
  test<std::unique_lock<std::timed_mutex>, std::chrono::steady_clock>();

  test<MyLock<std::mutex>, TestClock>();
  test<MyLock<std::mutex>, std::chrono::steady_clock>();

  test<MyLock<std::timed_mutex>, TestClock>();
  test<MyLock<std::timed_mutex>, std::chrono::steady_clock>();
  return 0;
}
