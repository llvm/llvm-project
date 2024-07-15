//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <mutex>

// template <class Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

// template<class _Mutex> lock_guard(lock_guard<_Mutex>)
//     -> lock_guard<_Mutex>;  // C++17

#include <mutex>
#include <cassert>
#include "test_macros.h"

struct Lock {
  bool locked = false;

  Lock() = default;
  ~Lock() { assert(!locked); }

  void lock() {
    assert(!locked);
    locked = true;
  }
  void unlock() {
    assert(locked);
    locked = false;
  }

  Lock(Lock const&)            = delete;
  Lock& operator=(Lock const&) = delete;
};

int main(int, char**) {
  Lock l;
  {
    std::lock_guard<Lock> lg(l);
    assert(l.locked);
  }
  assert(!l.locked);

#if TEST_STD_VER >= 17
  std::lock_guard lg(l);
  static_assert((std::is_same<decltype(l), std::lock_guard<decltype(l)>>::value), "");
#endif

  return 0;
}
