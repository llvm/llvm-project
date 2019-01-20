//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads

// <mutex>

// template <class Mutex> class unique_lock;

// explicit operator bool() const;

#include <mutex>
#include <cassert>

std::mutex m;

int main()
{
    std::unique_lock<std::mutex> lk0;
    assert(static_cast<bool>(lk0) == false);
    std::unique_lock<std::mutex> lk1(m);
    assert(static_cast<bool>(lk1) == true);
    lk1.unlock();
    assert(static_cast<bool>(lk1) == false);
}
