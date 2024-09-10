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

// template<class Lock, class Predicate>
//   bool wait(Lock& lock, stop_token stoken, Predicate pred);

#include <cassert>
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
  // stop_requested before hand
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};
    ss.request_stop();

    // [Note 1: The returned value indicates whether the predicate evaluated to true regardless of whether there was a stop request.]
    std::same_as<bool> auto r1 = cv.wait(lock, ss.get_token(), []() { return false; });
    assert(!r1);

    std::same_as<bool> auto r2 = cv.wait(lock, ss.get_token(), []() { return true; });
    assert(r2);

    // Postconditions: lock is locked by the calling thread.
    assert(lock.owns_lock());
  }

  // no stop request
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};
    std::same_as<bool> auto r1 = cv.wait(lock, ss.get_token(), []() { return true; });
    assert(r1);

    bool flag   = false;
    auto thread = support::make_test_thread([&]() {
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
      std::unique_lock<Mutex> lock2{mutex};
      flag = true;
      cv.notify_all();
    });

    std::same_as<bool> auto r2 = cv.wait(lock, ss.get_token(), [&]() { return flag; });
    assert(flag);
    assert(r2);
    thread.join();

    assert(lock.owns_lock());
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
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
    });

    std::same_as<bool> auto r = cv.wait(lock, ss.get_token(), [&]() {
      start.store(true);
      start.notify_all();
      return false;
    });
    assert(!r);
    done = true;
    thread.join();

    assert(lock.owns_lock());
  }

  // #76807 Hangs in std::condition_variable_any when used with std::stop_token
  {
    class MyThread {
    public:
      MyThread() {
        thread_ = support::make_test_jthread([this](std::stop_token st) {
          while (!st.stop_requested()) {
            std::unique_lock lock{m_};
            cv_.wait(lock, st, [] { return false; });
          }
        });
      }

    private:
      std::mutex m_;
      std::condition_variable_any cv_;
      std::jthread thread_;
    };

    [[maybe_unused]] MyThread my_thread;
  }

  // request_stop potentially in-between check and wait
  {
    std::stop_source ss;
    std::condition_variable_any cv;
    Mutex mutex;
    Lock lock{mutex};

    std::atomic_bool pred_started        = false;
    std::atomic_bool request_stop_called = false;
    auto thread                          = support::make_test_thread([&]() {
      pred_started.wait(false);
      ss.request_stop();
      request_stop_called.store(true);
      request_stop_called.notify_all();
    });

    std::same_as<bool> auto r = cv.wait(lock, ss.get_token(), [&]() {
      pred_started.store(true);
      pred_started.notify_all();
      request_stop_called.wait(false);
      return false;
    });
    assert(!r);
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
      cv.wait(lock, ss.get_token(), []() -> bool { throw 5; });
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
