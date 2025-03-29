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

// template <class Mutex> class shared_lock;

// void lock();

#include <atomic>
#include <cassert>
#include <mutex> // std::defer_lock
#include <shared_mutex>
#include <system_error>
#include <thread>
#include <vector>

#include "make_test_thread.h"
#include "test_macros.h"

struct Monitor {
  bool lock_shared_called   = false;
  bool unlock_shared_called = false;
};

struct TrackedMutex {
  Monitor* monitor = nullptr;

  void lock_shared() {
    if (monitor != nullptr)
      monitor->lock_shared_called = true;
  }
  void unlock_shared() {
    if (monitor != nullptr)
      monitor->unlock_shared_called = true;
  }
};

template <class Mutex>
void test() {
  // Basic sanity test
  {
    Mutex mutex;
    std::vector<std::thread> threads;
    std::atomic<bool> ready(false);
    for (int i = 0; i != 5; ++i) {
      threads.push_back(support::make_test_thread([&] {
        while (!ready) {
          // spin
        }

        std::shared_lock<Mutex> lock(mutex, std::defer_lock);
        lock.lock();
        assert(lock.owns_lock());
      }));
    }

    ready = true;
    for (auto& t : threads)
      t.join();
  }

  // Try locking the same shared_lock again in the same thread. This should throw an exception.
  {
    Mutex mutex;
    std::shared_lock<Mutex> lock(mutex, std::defer_lock);
    lock.lock();
    assert(lock.owns_lock());
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
      lock.lock();
      assert(false);
    } catch (std::system_error const& e) {
      assert(e.code() == std::errc::resource_deadlock_would_occur);
    }
#endif
  }

  // Try locking a shared_lock that isn't associated to any mutex. This should throw an exception.
  {
    std::shared_lock<Mutex> lock; // no associated mutex
#ifndef TEST_HAS_NO_EXCEPTIONS
    try {
      lock.lock();
      assert(false);
    } catch (std::system_error const& e) {
      assert(e.code() == std::errc::operation_not_permitted);
    }
#endif
  }
}

int main(int, char**) {
#if TEST_STD_VER >= 17
  test<std::shared_mutex>();
#endif
  test<std::shared_timed_mutex>();
  test<TrackedMutex>();

  // Use shared_lock with a dummy mutex class that tracks whether each
  // operation has been called or not.
  {
    Monitor monitor;
    TrackedMutex mutex{&monitor};

    std::shared_lock<TrackedMutex> lock(mutex, std::defer_lock);
    lock.lock();
    assert(monitor.lock_shared_called);
    assert(lock.owns_lock());

    lock.unlock();
    assert(monitor.unlock_shared_called);
  }

  return 0;
}
