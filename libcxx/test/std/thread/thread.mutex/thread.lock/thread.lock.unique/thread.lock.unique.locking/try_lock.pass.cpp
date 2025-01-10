//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// bool try_lock();

#include <cassert>
#include <mutex>
#include <system_error>

#include "test_macros.h"
#include "checking_mutex.h"

int main(int, char**) {
  checking_mutex mux;

  std::unique_lock<checking_mutex> lock(mux, std::defer_lock_t());
  assert(lock.try_lock());
  assert(mux.current_state == checking_mutex::locked_via_try_lock);
  assert(lock.owns_lock());

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    mux.last_try = checking_mutex::none;
    TEST_IGNORE_NODISCARD lock.try_lock();
    assert(false);
  } catch (std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == std::errc::resource_deadlock_would_occur);
  }
#endif

  lock.unlock();
  mux.reject = true;

  assert(!lock.try_lock());
  assert(mux.last_try == checking_mutex::locked_via_try_lock);

  assert(!lock.owns_lock());
  lock.release();

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    mux.last_try = checking_mutex::none;
    (void)lock.try_lock();
    assert(false);
  } catch (std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == std::errc::operation_not_permitted);
  }
#endif

  return 0;
}
