//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <mutex>

// template <class Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

// template<class _Mutex> lock_guard(lock_guard<_Mutex>)
//     -> lock_guard<_Mutex>;  // C++17

#include <mutex>
#include <cassert>

#include "test_macros.h"
#include "types.h"

int main(int, char**) {
  MyMutex m;
  {
    std::lock_guard<MyMutex> lg(m);
    assert(m.locked);
  }

  m.lock();
  m.unlock();

#if TEST_STD_VER >= 17
  std::lock_guard lg(m);
  static_assert((std::is_same<decltype(lg), std::lock_guard<decltype(m)>>::value), "");
#endif

  return 0;
}
