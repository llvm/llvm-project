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

// void lock();

#include <mutex>
#include <atomic>
#include <cassert>
#include <thread>
#include <vector>

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
  // Lock a mutex that is not locked yet. This should succeed.
  {
    std::recursive_mutex m;
    m.lock();
    m.unlock();
  }

  // Lock a mutex that is already locked by this thread. This should succeed and the mutex should only
  // be unlocked after a matching number of calls to unlock() on the same thread.
  {
    std::recursive_mutex m;
    int lock_count = 0;
    for (int i = 0; i != 10; ++i) {
      m.lock();
      ++lock_count;
    }
    while (lock_count != 0) {
      assert(!is_lockable(m));
      m.unlock();
      --lock_count;
    }
    assert(is_lockable(m));
  }

  // Lock a mutex that is already locked by another thread. This should block until it is unlocked.
  {
    std::atomic<bool> ready(false);
    std::recursive_mutex m;
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
    // We instead signal it before we unlock, which means that it's technically possible for
    // the thread to take the lock while main is still holding it yet for the test to still pass.
    is_locked_from_main = false;
    m.unlock();

    t.join();
  }

  // Make sure that at most one thread can acquire the mutex concurrently.
  {
    std::atomic<int> counter(0);
    std::recursive_mutex mutex;

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
