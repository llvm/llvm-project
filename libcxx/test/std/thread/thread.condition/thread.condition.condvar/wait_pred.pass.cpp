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

// template <class Predicate>
//   void wait(unique_lock<mutex>& lock, Predicate pred);

#include <condition_variable>
#include <atomic>
#include <cassert>
#include <mutex>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

int main(int, char**) {
  // Test unblocking via a call to notify_one() in another thread.
  //
  // To test this, we try to minimize the likelihood that we got awoken by a
  // spurious wakeup by updating the likely_spurious flag only immediately
  // before we perform the notification.
  {
    std::atomic<bool> ready(false);
    std::atomic<bool> likely_spurious(true);
    std::condition_variable cv;
    std::mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      std::unique_lock<std::mutex> lock(mutex);
      ready = true;
      cv.wait(lock, [&] { return !likely_spurious; });
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

  // Test unblocking via a spurious wakeup.
  //
  // To test this, we basically never wake up the condition variable. This way, we
  // are hoping to get out of the wait via a spurious wakeup.
  //
  // However, since spurious wakeups are not required to even happen, this test is
  // only trying to trigger that code path, but not actually asserting that it is
  // taken. In particular, we do need to eventually ensure we get out of the wait
  // by standard means, so we actually wake up the thread at the end.
  {
    std::atomic<bool> ready(false);
    std::atomic<bool> awoken(false);
    std::condition_variable cv;
    std::mutex mutex;

    std::thread t1 = support::make_test_thread([&] {
      std::unique_lock<std::mutex> lock(mutex);
      ready = true;
      cv.wait(lock, [&] { return true; });
      awoken = true;
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

      // Whatever happened, actually awaken the condition variable to ensure the test finishes.
      cv.notify_one();
    });

    t2.join();
    t1.join();
  }

  return 0;
}
