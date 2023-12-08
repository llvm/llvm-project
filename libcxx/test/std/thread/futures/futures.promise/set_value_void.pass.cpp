//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// <future>

// class promise<R>

// void promise<void>::set_value();

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef void T;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        p.set_value();
        f.get();
        try
        {
            p.set_value();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::promise_already_satisfied));
        }
    }

  return 0;
}
