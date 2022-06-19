//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <mutex>

// template <class Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

// template<class _Mutex> lock_guard(lock_guard<_Mutex>)
//     -> lock_guard<_Mutex>;  // C++17

#include <mutex>
#include <cstdlib>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

std::mutex m;

void do_try_lock() {
  assert(m.try_lock() == false);
}

int main(int, char**) {
  {
    std::lock_guard<std::mutex> lg(m);
    std::thread t = support::make_test_thread(do_try_lock);
    t.join();
  }

  m.lock();
  m.unlock();

#if TEST_STD_VER >= 17
  std::lock_guard lg(m);
  static_assert((std::is_same<decltype(lg), std::lock_guard<decltype(m)>>::value), "" );
#endif

  return 0;
}
