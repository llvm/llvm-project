//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03, c++11

// <shared_mutex>

// template <class Mutex> class shared_lock;

// explicit operator bool() const noexcept;

#include <shared_mutex>
#include <cassert>

#include "test_macros.h"

struct M {
    void lock_shared() {}
    void unlock_shared() {}
};

int main(int, char**)
{
    static_assert(std::is_constructible<bool, std::shared_lock<M>>::value, "");
    static_assert(!std::is_convertible<std::shared_lock<M>, bool>::value, "");

    M m;
    std::shared_lock<M> lk0;
    assert(static_cast<bool>(lk0) == false);
    std::shared_lock<M> lk1(m);
    assert(static_cast<bool>(lk1) == true);
    lk1.unlock();
    assert(static_cast<bool>(lk1) == false);
    ASSERT_NOEXCEPT(static_cast<bool>(lk0));

    return 0;
}
