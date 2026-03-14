//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class unique_lock;

// mutex_type *mutex() const;

#include <cassert>
#include <memory>
#include <mutex>

#include "checking_mutex.h"
#include "test_macros.h"

#if TEST_STD_VER >= 11
static_assert(noexcept(std::declval<std::unique_lock<checking_mutex>&>().mutex()), "");
#endif

int main(int, char**) {
  checking_mutex mux;
  const std::unique_lock<checking_mutex> lock0; // Make sure `mutex()` is `const`
  static_assert(std::is_same<decltype(lock0.mutex()), checking_mutex*>::value, "");
  assert(lock0.mutex() == nullptr);
  std::unique_lock<checking_mutex> lock1(mux);
  assert(lock1.mutex() == std::addressof(mux));
  lock1.unlock();
  assert(lock1.mutex() == std::addressof(mux));

  return 0;
}
