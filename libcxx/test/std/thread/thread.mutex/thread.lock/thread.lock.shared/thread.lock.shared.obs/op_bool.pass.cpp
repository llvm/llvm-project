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

// dylib support for shared_mutex was added in macosx10.12
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.11
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.10
// XFAIL: use_system_cxx_lib && x86_64-apple-macosx10.9

// <shared_mutex>

// template <class Mutex> class shared_lock;

// explicit operator bool() const noexcept;

#include <shared_mutex>
#include <cassert>

#include "test_macros.h"

std::shared_timed_mutex m;

int main(int, char**)
{
    std::shared_lock<std::shared_timed_mutex> lk0;
    assert(static_cast<bool>(lk0) == false);
    std::shared_lock<std::shared_timed_mutex> lk1(m);
    assert(static_cast<bool>(lk1) == true);
    lk1.unlock();
    assert(static_cast<bool>(lk1) == false);
    static_assert(noexcept(static_cast<bool>(lk0)), "explicit operator bool() must be noexcept");

  return 0;
}
