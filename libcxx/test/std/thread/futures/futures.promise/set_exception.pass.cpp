//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-no-exceptions
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++98, c++03

// <future>

// class promise<R>

// void set_exception(exception_ptr p);

#include <future>
#include <cassert>

int main(int, char**)
{
    {
        typedef int T;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_exception(std::make_exception_ptr(3));
        try
        {
            f.get();
            assert(false);
        }
        catch (int i)
        {
            assert(i == 3);
        }
        try
        {
            p.set_exception(std::make_exception_ptr(3));
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
        }
    }

  return 0;
}
