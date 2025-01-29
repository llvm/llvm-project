//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads

// <mutex>

// class mutex;

// mutex() noexcept;

#include <mutex>
#include <cassert>
#include <type_traits>

static_assert(std::is_nothrow_default_constructible<std::mutex>::value, "");

int main(int, char**) {
  // The mutex is unlocked after default construction
  {
    std::mutex m;
    assert(m.try_lock());
    m.unlock();
  }

  return 0;
}
