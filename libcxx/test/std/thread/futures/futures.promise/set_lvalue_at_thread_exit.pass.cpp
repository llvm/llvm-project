//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03

// <future>

// class promise<R>

// void promise<R&>::set_value_at_thread_exit(R& r);

#include <future>
#include <memory>
#include <cassert>

#include "make_test_thread.h"
#include "test_macros.h"

int i = 0;

void func(std::promise<int&> p)
{
    p.set_value_at_thread_exit(i);
    i = 4;
}

int main(int, char**)
{
    {
        std::promise<int&> p;
        std::future<int&> f = p.get_future();
        support::make_test_thread(func, std::move(p)).detach();
        assert(f.get() == 4);
    }

  return 0;
}
