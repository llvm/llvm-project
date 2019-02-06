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

// bool owns_lock() const;

#include <mutex>
#include <cassert>

std::mutex m;

int main()
{
    std::unique_lock<std::mutex> lk0;
    assert(lk0.owns_lock() == false);
    std::unique_lock<std::mutex> lk1(m);
    assert(lk1.owns_lock() == true);
    lk1.unlock();
    assert(lk1.owns_lock() == false);
}
