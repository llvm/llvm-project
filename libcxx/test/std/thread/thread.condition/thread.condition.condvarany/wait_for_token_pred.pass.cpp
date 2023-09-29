//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: libcpp-has-no-experimental-stop_token
// XFAIL: availability-synchronization_library-missing

// <condition_variable>

// class condition_variable_any;

// template<class Lock, class Rep, class Period, class Predicate>
//   bool wait_for(Lock& lock, stop_token stoken,
//                 const chrono::duration<Rep, Period>& rel_time, Predicate pred);

#include <cassert>
#include <chrono>
#include <concepts>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <shared_mutex>
#include <stop_token>
#include <thread>

#include "make_test_thread.h"
#include "test_macros.h"

template <class Mutex, class Lock>
void test() {
  using namespace std::chrono_literals;

  // stop_requested before hand
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};
    ss.request_stop();

    // [Note 4: The returned value indicates whether the predicate evaluated to true
    // regardless of whether the timeout was triggered or a stop request was made.]
    std::same_as<bool> auto r1 = cv.wait_for(lock, ss.get_token(), -1h, []() { return false; });
    assert(!r1);

    std::same_as<bool> auto r2 = cv.wait_for(lock, ss.get_token(), 1h, []() { return false; });
    assert(!r2);

    std::same_as<bool> auto r3 = cv.wait_for(lock, ss.get_token(), -1h, []() { return true; });
    assert(r3);

    std::same_as<bool> auto r4 = cv.wait_for(lock, ss.get_token(), 1h, []() { return true; });
    assert(r4);

    // Postconditions: lock is locked by the calling thread.
    assert(lock.owns_lock());
  }

  // no stop request, pred was true
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};

    std::same_as<bool> auto r1 = cv.wait_for(lock, ss.get_token(), -1h, []() { return true; });
    assert(r1);

    std::same_as<bool> auto r2 = cv.wait_for(lock, ss.get_token(), 1h, []() { return true; });
    assert(r2);
  }

  // no stop request, pred was false, abs_time was in the past
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};

    std::same_as<bool> auto r1 = cv.wait_for(lock, ss.get_token(), -1h, []() { return false; });
    assert(!r1);
  }

  // no stop request, pred was false until timeout
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};

    auto old_time = std::chrono::steady_clock::now();

    std::same_as<bool> auto r1 = cv.wait_for(lock, ss.get_token(), 2ms, [&]() { return false; });

    assert((std::chrono::steady_clock::now() - old_time) >= 2ms);
    assert(!r1);
  }

  // no stop request, pred was false, changed to true before timeout
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};

    bool flag   = false;
    auto thread = support::make_test_thread([&]() {
      std::this_thread::sleep_for(2ms);
      Lock lock2{mutex};
      flag = true;
      cv.notify_all();
    });

    std::same_as<bool> auto r1 = cv.wait_for(lock, ss.get_token(), 1h, [&]() { return flag; });
    assert(flag);
    assert(r1);

    thread.join();
  }

  // stop request comes while waiting
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};

    std::atomic_bool start = false;
    std::atomic_bool done  = false;
    auto thread            = support::make_test_thread([&]() {
      start.wait(false);
      ss.request_stop();

      while (!done) {
        cv.notify_all();
        std::this_thread::sleep_for(2ms);
      }
    });

    std::same_as<bool> auto r = cv.wait_for(lock, ss.get_token(), 1h, [&]() {
      start.store(true);
      start.notify_all();
      return false;
    });
    assert(!r);
    done = true;
    thread.join();

    assert(lock.owns_lock());
  }

#if !defined(TEST_HAS_NO_EXCEPTIONS)
  // Throws: Any exception thrown by pred.
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};

    try {
      cv.wait_for(lock, ss.get_token(), 1h, []() -> bool { throw 5; });
      assert(false);
    } catch (int i) {
      assert(i == 5);
    }
  }
#endif //!defined(TEST_HAS_NO_EXCEPTIONS)
}

int main(int, char**) {
  test<std::mutex, std::unique_lock<std::mutex>>();
  test<std::shared_mutex, std::shared_lock<std::shared_mutex>>();

  return 0;
}
