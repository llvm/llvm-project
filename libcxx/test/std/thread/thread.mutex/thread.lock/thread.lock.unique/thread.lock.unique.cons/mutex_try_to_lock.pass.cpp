//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock(mutex_type& m, try_to_lock_t);

#include <cassert>
#include <mutex>

#include "checking_mutex.h"

int main(int, char**) {
  checking_mutex mux;

  { // check successful lock
    mux.reject = false;
    std::unique_lock<checking_mutex> lock(mux, std::try_to_lock_t());
    assert(mux.current_state == checking_mutex::locked_via_try_lock);
    assert(lock.owns_lock());
  }
  assert(mux.current_state == checking_mutex::unlocked);

  { // check successful lock
    mux.reject = true;
    std::unique_lock<checking_mutex> lock(mux, std::try_to_lock_t());
    assert(mux.last_try == checking_mutex::locked_via_try_lock);
    assert(mux.current_state == checking_mutex::unlocked);
    assert(!lock.owns_lock());
  }
  assert(mux.current_state == checking_mutex::unlocked);

  return 0;
}
