//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// Make sure std::unique_lock works with std::mutex as expected.

#include <atomic>
#include <cassert>
#include <mutex>

#include "make_test_thread.h"

std::atomic<bool> keep_waiting;
std::atomic<bool> child_thread_locked;
std::mutex mux;
bool main_thread_unlocked  = false;
bool child_thread_unlocked = false;

void lock_thread() {
  std::unique_lock<std::mutex> lock(mux);
  assert(main_thread_unlocked);
  main_thread_unlocked  = false;
  child_thread_unlocked = true;
}

void try_lock_thread() {
  std::unique_lock<std::mutex> lock(mux, std::try_to_lock_t());
  assert(lock.owns_lock());
  child_thread_locked = true;

  while (keep_waiting)
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

  child_thread_unlocked = true;
}

int main(int, char**) {
  {
    mux.lock();
    std::thread t        = support::make_test_thread(lock_thread);
    main_thread_unlocked = true;
    mux.unlock();
    t.join();
    assert(child_thread_unlocked);
  }

  {
    child_thread_unlocked = false;
    child_thread_locked   = false;
    keep_waiting          = true;
    std::thread t         = support::make_test_thread(try_lock_thread);
    while (!child_thread_locked)
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    assert(!mux.try_lock());
    keep_waiting = false;
    t.join();
    assert(child_thread_unlocked);
  }

  return 0;
}
