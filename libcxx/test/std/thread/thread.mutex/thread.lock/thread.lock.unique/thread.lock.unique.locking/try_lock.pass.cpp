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

// bool try_lock();

#include <mutex>
#include <cassert>

#include "test_macros.h"

bool try_lock_called = false;

struct mutex
{
    bool try_lock()
    {
        try_lock_called = !try_lock_called;
        return try_lock_called;
    }
    void unlock() {}
};

mutex m;

int main()
{
    std::unique_lock<mutex> lk(m, std::defer_lock);
    assert(lk.try_lock() == true);
    assert(try_lock_called == true);
    assert(lk.owns_lock() == true);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        TEST_IGNORE_NODISCARD lk.try_lock();
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EDEADLK);
    }
#endif
    lk.unlock();
    assert(lk.try_lock() == false);
    assert(try_lock_called == false);
    assert(lk.owns_lock() == false);
    lk.release();
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        TEST_IGNORE_NODISCARD lk.try_lock();
        assert(false);
    }
    catch (std::system_error& e)
    {
        assert(e.code().value() == EPERM);
    }
#endif
}
