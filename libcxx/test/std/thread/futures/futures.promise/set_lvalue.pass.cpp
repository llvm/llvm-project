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

// void promise<R&>::set_value(R& r);

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef int& T;
        int i = 3;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_value(i);
        int& j = f.get();
        assert(j == 3);
        ++i;
        assert(j == 4);
#ifndef TEST_HAS_NO_EXCEPTIONS
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

  return 0;
}
