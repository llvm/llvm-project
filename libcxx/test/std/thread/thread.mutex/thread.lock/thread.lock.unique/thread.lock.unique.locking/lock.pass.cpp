//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// void lock();

#include <cassert>
#include <mutex>
#include <system_error>

#include "checking_mutex.h"
#include "test_macros.h"

int main(int, char**) {
  checking_mutex mux;
  std::unique_lock<checking_mutex> lk(mux, std::defer_lock_t());
  assert(mux.last_try == checking_mutex::none);
  lk.lock();
  assert(mux.current_state == checking_mutex::locked_via_lock);
  mux.last_try = checking_mutex::none;

#ifndef TEST_HAS_NO_EXCEPTIONS
  try {
    mux.last_try = checking_mutex::none;
    lk.lock();
    assert(false);
  } catch (std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == std::errc::resource_deadlock_would_occur);
  }

  lk.unlock();
  lk.release();

  try {
    mux.last_try = checking_mutex::none;
    lk.lock();
    assert(false);
  } catch (std::system_error& e) {
    assert(mux.last_try == checking_mutex::none);
    assert(e.code() == std::errc::operation_not_permitted);
  }
#endif

  return 0;
}
