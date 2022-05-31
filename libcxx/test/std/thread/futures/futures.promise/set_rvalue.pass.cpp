//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: no-threads, no-exceptions

// <future>

// class promise<R>

// void promise::set_value(R&& r);

#include <future>
#include <memory>
#include <cassert>

#include "test_macros.h"

struct A
{
    A() {}
    A(const A&) = delete;
    A(A&&) {throw 9;}
};

int main(int, char**)
{
    {
        typedef std::unique_ptr<int> T;
        T i(new int(3));
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_value(std::move(i));
        assert(*f.get() == 3);
        try
        {
            p.set_value(std::move(i));
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
        }
    }
    {
        typedef A T;
        T i;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        try
        {
            p.set_value(std::move(i));
            assert(false);
        }
        catch (int j)
        {
            assert(j == 9);
        }
    }

  return 0;
}
