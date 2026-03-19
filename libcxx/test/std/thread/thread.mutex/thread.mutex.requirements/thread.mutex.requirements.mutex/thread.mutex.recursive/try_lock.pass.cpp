//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads

// <mutex>

// class recursive_mutex;

// bool try_lock();

#include <mutex>
#include <atomic>
#include <cassert>
#include <chrono>
#include <thread>

#include "make_test_thread.h"

bool is_lockable(std::recursive_mutex& m) {
  bool did_lock;
  std::thread t = support::make_test_thread([&] {
    did_lock = m.try_lock();
    if (did_lock)
      m.unlock(); // undo side effects
  });
  t.join();

  return did_lock;
}

int main(int, char**) {
  // Try to lock a mutex that is not locked yet. This should succeed.
  {
    std::recursive_mutex m;
    bool succeeded = m.try_lock();
    assert(succeeded);
    m.unlock();
  }

  // Try to lock a mutex that is already locked by this thread. This should succeed and the mutex should only
  // be unlocked after a matching number of calls to unlock() on the same thread.
  {
    std::recursive_mutex m;
    int lock_count = 0;
    for (int i = 0; i != 10; ++i) {
      assert(m.try_lock());
      ++lock_count;
    }
    while (lock_count != 0) {
      assert(!is_lockable(m));
      m.unlock();
      --lock_count;
    }
    assert(is_lockable(m));
  }

  // Try to lock a mutex that is already locked by another thread. This should fail.
  {
    std::recursive_mutex m;
    m.lock();

    std::thread t = support::make_test_thread([&] {
      for (int i = 0; i != 10; ++i) {
        bool succeeded = m.try_lock();
        assert(!succeeded);
      }
    });
    t.join();

    m.unlock();
  }

  return 0;
}
