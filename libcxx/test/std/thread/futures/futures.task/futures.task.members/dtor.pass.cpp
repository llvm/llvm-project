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

// ~packaged_task();

#include <future>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

class A
{
    long data_;

public:
    explicit A(long i) : data_(i) {}

    long operator()(long i, long j) const {return data_ + i + j;}
};

void func(std::packaged_task<double(int, char)>)
{
}

void func2(std::packaged_task<double(int, char)> p)
{
    p(3, 'a');
}

int main(int, char**)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        support::make_test_thread(func, std::move(p)).detach();
        try
        {
            double i = f.get();
            ((void)i); // Prevent unused warning
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::broken_promise));
        }
    }
#endif
    {
        std::packaged_task<double(int, char)> p(A(5));
        std::future<double> f = p.get_future();
        support::make_test_thread(func2, std::move(p)).detach();
        assert(f.get() == 105.0);
    }

  return 0;
}
