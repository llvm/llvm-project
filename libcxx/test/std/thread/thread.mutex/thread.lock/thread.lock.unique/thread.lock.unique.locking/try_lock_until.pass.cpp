//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: c++03

// <mutex>

// template <class Mutex> class unique_lock;

// template <class Clock, class Duration>
//   bool try_lock_until(const chrono::time_point<Clock, Duration>& abs_time);

#include <cassert>
#include <chrono>
#include <mutex>
#include <system_error>

#include "test_macros.h"
#include "../types.h"

MyTimedMutex m;

int main(int, char**) {
  typedef std::chrono::system_clock Clock;
  std::unique_lock<MyTimedMutex> lk(m, std::defer_lock);
  assert(lk.try_lock_until(Clock::now()) == true);
  assert(m.try_lock_until_called == true);
  assert(lk.owns_lock() == true);
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    TEST_IGNORE_NODISCARD lk.try_lock_until(Clock::now());
    assert(false);
  } catch (std::system_error& e) {
    assert(e.code() == std::errc::resource_deadlock_would_occur);
  }
#endif
  lk.unlock();
  assert(lk.try_lock_until(Clock::now()) == false);
  assert(m.try_lock_until_called == false);
  assert(lk.owns_lock() == false);
  lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    TEST_IGNORE_NODISCARD lk.try_lock_until(Clock::now());
    assert(false);
  } catch (std::system_error& e) {
    assert(e.code() == std::errc::operation_not_permitted);
  }
#endif

  return 0;
}
