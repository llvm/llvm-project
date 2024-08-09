//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Rep, class Period>
// unique_lock::unique_lock(mutex_type& m, const chrono::duration<Rep, Period>& rel_time);

#include <cassert>
#include <chrono>
#include <mutex>

#include "checking_mutex.h"

int main(int, char**) {
  checking_mutex mux;
  { // check successful lock
    mux.reject = false;
    std::unique_lock<checking_mutex> lock(mux, std::chrono::seconds());
    assert(mux.current_state == checking_mutex::locked_via_try_lock_for);
    assert(lock.owns_lock());
  }
  assert(mux.current_state == checking_mutex::unlocked);

  { // check unsuccessful lock
    mux.reject = true;
    std::unique_lock<checking_mutex> lock(mux, std::chrono::seconds());
    assert(mux.current_state == checking_mutex::unlocked);
    assert(mux.last_try == checking_mutex::locked_via_try_lock_for);
    assert(!lock.owns_lock());
  }
  assert(mux.current_state == checking_mutex::unlocked);

  return 0;
}
