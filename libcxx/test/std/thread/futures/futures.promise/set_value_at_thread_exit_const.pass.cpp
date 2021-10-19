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

// void promise::set_value_at_thread_exit(const R& r);

#include <future>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

void func(std::promise<int> p)
{
    const int i = 5;
    p.set_value_at_thread_exit(i);
}

int main(int, char**)
{
    {
        std::promise<int> p;
        std::future<int> f = p.get_future();
        support::make_test_thread(func, std::move(p)).detach();
        assert(f.get() == 5);
    }

  return 0;
}
