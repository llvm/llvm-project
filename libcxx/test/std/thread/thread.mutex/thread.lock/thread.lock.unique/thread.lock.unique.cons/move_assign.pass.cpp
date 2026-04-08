//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: FROZEN-CXX03-HEADERS-FIXME

// <mutex>

// template <class Mutex> class unique_lock;

// unique_lock& operator=(unique_lock&& u) noexcept;

#include <cassert>
#include <memory>
#include <mutex>
#include <type_traits>

#include "checking_mutex.h"

int main(int, char**) {
  {
    checking_mutex m0;
    checking_mutex m1;
    std::unique_lock<checking_mutex> lk0(m0);
    std::unique_lock<checking_mutex> lk1(m1);

    // Test self move assignment for lk0.
    lk0 = std::move(lk0);
    assert(lk0.mutex() == std::addressof(m0));
    assert(lk0.owns_lock() == true);

    auto& result = (lk1 = std::move(lk0));

    assert(&result == &lk1);
    assert(lk1.mutex() == std::addressof(m0));
    assert(lk1.owns_lock());
    assert(lk0.mutex() == nullptr);
    assert(lk0.owns_lock() == false);

    static_assert(std::is_nothrow_move_assignable<std::unique_lock<checking_mutex> >::value, "");
  }

  {
    // Test self move-assignment (LWG4172)
    checking_mutex m0;
    std::unique_lock<checking_mutex> lk0(m0);
    lk0 = std::move(lk0);
    assert(lk0.mutex() == std::addressof(m0));
    assert(lk0.owns_lock() == true);
  }

  return 0;
}
