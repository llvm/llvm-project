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

// class packaged_task<R(ArgTypes...)>

// future<R> get_future();

#include <future>
#include <cassert>

#include "test_macros.h"

class A
{
    long data_;

public:
    explicit A(long i) : data_(i) {}

    long operator()(long i, long j) const {return data_ + i + j;}
};

int main(int, char**)
{
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        p(3, 'a');
        assert(f.get() == 105.0);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::packaged_task<double(int, char)> p(A(5));
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
        std::packaged_task<double(int, char)> p;
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
