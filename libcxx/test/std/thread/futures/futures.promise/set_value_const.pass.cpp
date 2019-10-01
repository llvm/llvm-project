//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <future>

// class promise<R>

// void promise::set_value(const R& r);

#include <future>
#include <cassert>

#include "test_macros.h"

struct A
{
    A() {}
    A(const A&) {
        TEST_THROW(10);
    }
};

int main(int, char**)
{
    {
        typedef int T;
        T i = 3;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_value(i);
        ++i;
        assert(f.get() == 3);
#ifndef TEST_HAS_NO_EXCEPTIONS
        --i;
        try
        {
            p.set_value(i);
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
        }
#endif
    }
    {
        typedef A T;
        T i;
        std::promise<T> p;
        std::future<T> f = p.get_future();
#ifndef TEST_HAS_NO_EXCEPTIONS
        try
        {
            p.set_value(i);
            assert(false);
        }
        catch (int j)
        {
            assert(j == 10);
        }
#endif
    }

  return 0;
}
