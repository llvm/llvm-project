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

// void promise::set_exception_at_thread_exit(exception_ptr p);

#include <future>
#include <cassert>

#include "test_macros.h"

void func(std::promise<int> p)
{
    p.set_exception_at_thread_exit(std::make_exception_ptr(3));
}

int main(int, char**)
{
    {
        typedef int T;
        std::promise<T> p;
        std::future<T> f = p.get_future();
        std::thread(func, std::move(p)).detach();
        try
        {
            f.get();
            assert(false);
        }
        catch (int i)
        {
            assert(i == 3);
        }
    }

  return 0;
}
