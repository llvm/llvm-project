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
#include <cassert>
#include "test_macros.h"

struct TestMutex {
    bool locked = false;
    TestMutex() = default;
    ~TestMutex() { assert(!locked); }

    void lock() { assert(!locked); locked = true; }
    bool try_lock() { if (locked) return false; locked = true; return true; }
    void unlock() { assert(locked); locked = false; }

    TestMutex(TestMutex const&) = delete;
    TestMutex& operator=(TestMutex const&) = delete;
};

int main(int, char**) {
  TestMutex m;
  {
    std::lock_guard<TestMutex> lg(m);
    assert(m.locked);
  }
  assert(!m.locked);

#if TEST_STD_VER >= 17
  std::lock_guard lg(m);
  static_assert((std::is_same<decltype(lg), std::lock_guard<decltype(m)>>::value), "" );
#endif

  return 0;
}
