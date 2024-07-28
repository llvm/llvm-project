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

// class condition_variable;

// template <class Clock, class Duration, class Predicate>
//     bool
//     wait_until(unique_lock<mutex>& lock,
//                const chrono::time_point<Clock, Duration>& abs_time,
//                Predicate pred);

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

template <class Clock>
void test() {
  // Test unblocking via a call to notify_one() in another thread.
  //
  // To test this, we set a very long timeout in wait_until() and we try to minimize
  // the likelihood that we got awoken by a spurious wakeup by updating the
  // likely_spurious flag only immediately before we perform the notification.
  {
    std::atomic<bool> ready(false);
    std::atomic<bool> likely_spurious(true);
    auto timeout = Clock::now() + std::chrono::seconds(3600);
    std::condition_variable cv;
    std::mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      std::unique_lock<std::mutex> lock(mutex);
      ready       = true;
      bool result = cv.wait_until(lock, timeout, [&] { return !likely_spurious; });
      assert(result); // return value should be true since we didn't time out
      assert(Clock::now() < timeout);
    });

    std::thread t2 = support::make_test_thread([&] {
      while (!ready) {
        // spin
      }

      // Acquire the same mutex as t1. This ensures that the condition variable has started
      // waiting (and hence released that mutex).
      std::unique_lock<std::mutex> lock(mutex);

      likely_spurious = false;
      lock.unlock();
      cv.notify_one();
    });

    t2.join();
    t1.join();
  }

  // Test unblocking via a timeout.
  //
  // To test this, we create a thread that waits on a condition variable with a certain
  // timeout, and we never awaken it. The "stop waiting" predicate always returns false,
  // which means that we can't get out of the wait via a spurious wakeup.
  {
    auto timeout = Clock::now() + std::chrono::milliseconds(250);
    std::condition_variable cv;
    std::mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      std::unique_lock<std::mutex> lock(mutex);
      bool result = cv.wait_until(lock, timeout, [] { return false; }); // never stop waiting (until timeout)
      assert(!result); // return value should be false since the predicate returns false after the timeout
      assert(Clock::now() >= timeout);
    });

    t1.join();
  }

  // Test unblocking via a spurious wakeup.
  //
  // To test this, we set a fairly long timeout in wait_until() and we basically never
  // wake up the condition variable. This way, we are hoping to get out of the wait
  // via a spurious wakeup.
  //
  // However, since spurious wakeups are not required to even happen, this test is
  // only trying to trigger that code path, but not actually asserting that it is
  // taken. In particular, we do need to eventually ensure we get out of the wait
  // by standard means, so we actually wake up the thread at the end.
  {
    std::atomic<bool> ready(false);
    std::atomic<bool> awoken(false);
    auto timeout = Clock::now() + std::chrono::seconds(3600);
    std::condition_variable cv;
    std::mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      std::unique_lock<std::mutex> lock(mutex);
      ready       = true;
      bool result = cv.wait_until(lock, timeout, [&] { return true; });
      awoken      = true;
      assert(result);                 // return value should be true since we didn't time out
      assert(Clock::now() < timeout); // can technically fail if t2 never executes and we timeout, but very unlikely
    });

    std::thread t2 = support::make_test_thread([&] {
      while (!ready) {
        // spin
      }

      // Acquire the same mutex as t1. This ensures that the condition variable has started
      // waiting (and hence released that mutex).
      std::unique_lock<std::mutex> lock(mutex);
      lock.unlock();

      // Give some time for t1 to be awoken spuriously so that code path is used.
      std::this_thread::sleep_for(std::chrono::seconds(1));

      // We would want to assert that the thread has been awoken after this time,
      // however nothing guarantees us that it ever gets spuriously awoken, so
      // we can't really check anything. This is still left here as documentation.
      bool woke = awoken.load();
      assert(woke || !woke);

      // Whatever happened, actually awaken the condition variable to ensure the test
      // doesn't keep running until the timeout.
      cv.notify_one();
    });

    t2.join();
    t1.join();
  }
}

int main(int, char**) {
  test<TestClock>();
  test<std::chrono::steady_clock>();
  return 0;
}
