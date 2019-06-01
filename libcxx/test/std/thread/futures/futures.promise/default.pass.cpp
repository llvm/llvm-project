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

// promise();

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        std::promise<int> p;
        std::future<int> f = p.get_future();
        assert(f.valid());
    }
    {
        std::promise<int&> p;
        std::future<int&> f = p.get_future();
        assert(f.valid());
    }
    {
        std::promise<void> p;
        std::future<void> f = p.get_future();
        assert(f.valid());
    }

  return 0;
}
