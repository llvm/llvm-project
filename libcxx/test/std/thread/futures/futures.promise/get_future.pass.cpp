//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <future>

// class promise<R>

// future<R> get_future();

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::promise<double> p;
        std::future<double> f = p.get_future();
        p.set_value(105.5);
        assert(f.get() == 105.5);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::promise<double> p;
        std::future<double> f = p.get_future();
        try
        {
            f = p.get_future();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() ==  make_error_code(std::future_errc::future_already_retrieved));
        }
    }
    {
        std::promise<double> p;
        std::promise<double> p0 = std::move(p);
        try
        {
            std::future<double> f = p.get_future();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() ==  make_error_code(std::future_errc::no_state));
        }
    }
#endif

  return 0;
}
