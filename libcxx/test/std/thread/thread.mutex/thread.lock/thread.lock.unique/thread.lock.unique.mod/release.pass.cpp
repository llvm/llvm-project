//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// mutex_type* release() noexcept;

#include <cassert>
#include <memory>
#include <mutex>

#include "checking_mutex.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
static_assert(noexcept(std::declval<std::unique_lock<checking_mutex>&>().release()), "");
#endif

int main(int, char**) {
  checking_mutex mux;
  std::unique_lock<checking_mutex> lock(mux);
  assert(lock.mutex() == std::addressof(mux));
  assert(lock.owns_lock());

  assert(mux.current_state == checking_mutex::locked_via_lock);

  assert(lock.release() == std::addressof(mux));
  assert(lock.mutex() == nullptr);
  assert(!lock.owns_lock());
  assert(mux.last_try == checking_mutex::locked_via_lock);
  assert(mux.current_state == checking_mutex::locked_via_lock);
  mux.unlock();

  return 0;
}
