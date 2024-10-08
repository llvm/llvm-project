//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// bool owns_lock() const;

#include <cassert>
#include <mutex>

#include "checking_mutex.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
static_assert(noexcept(std::declval<std::unique_lock<checking_mutex>&>().owns_lock()), "");
#endif

int main(int, char**) {
  {
    checking_mutex mux;
    const std::unique_lock<checking_mutex> lock0; // Make sure `owns_lock()` is `const`
    assert(!lock0.owns_lock());
    std::unique_lock<checking_mutex> lock1(mux);
    assert(lock1.owns_lock());
    lock1.unlock();
    assert(!lock1.owns_lock());
  }

  return 0;
}
