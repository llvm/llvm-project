//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// class recursive_mutex;

// recursive_mutex();

#include <mutex>
#include <cassert>
#include <type_traits>

int main(int, char**) {
  // The mutex is unlocked after default construction
  {
    std::recursive_mutex m;
    assert(m.try_lock());
    m.unlock();
  }

  return 0;
}
