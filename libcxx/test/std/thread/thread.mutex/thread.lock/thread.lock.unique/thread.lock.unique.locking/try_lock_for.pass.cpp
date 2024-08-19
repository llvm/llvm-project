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

// template <class Rep, class Period>
//   bool try_lock_for(const chrono::duration<Rep, Period>& rel_time);

#include <cassert>
#include <mutex>
#include <system_error>

#include "test_macros.h"
#include "../types.h"

MyTimedMutex m;

int main(int, char**) {
  using ms = std::chrono::milliseconds;
  std::unique_lock<MyTimedMutex> lk(m, std::defer_lock);
  assert(lk.try_lock_for(ms(5)) == true);
  assert(m.try_lock_for_called == true);
  assert(lk.owns_lock() == true);
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    TEST_IGNORE_NODISCARD lk.try_lock_for(ms(5));
    assert(false);
  } catch (std::system_error& e) {
    assert(e.code() == std::errc::resource_deadlock_would_occur);
  }
#endif
  lk.unlock();
  assert(lk.try_lock_for(ms(5)) == false);
  assert(m.try_lock_for_called == false);
  assert(lk.owns_lock() == false);
  lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    TEST_IGNORE_NODISCARD lk.try_lock_for(ms(5));
    assert(false);
  } catch (std::system_error& e) {
    assert(e.code() == std::errc::operation_not_permitted);
  }
#endif

  return 0;
}
