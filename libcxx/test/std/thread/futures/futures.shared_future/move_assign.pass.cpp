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

// class shared_future<R>

// shared_future& operator=(shared_future&& rhs);

#include <future>
#include <cassert>

#include "test_macros.h"

int main(int, char**)
{
    {
        typedef int T;
        std::promise<T> p;
        std::shared_future<T> f0 = p.get_future();
        std::shared_future<T> f;
        f = std::move(f0);
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef int T;
        std::shared_future<T> f0;
        std::shared_future<T> f;
        f = std::move(f0);
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef int& T;
        std::promise<T> p;
        std::shared_future<T> f0 = p.get_future();
        std::shared_future<T> f;
        f = std::move(f0);
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef int& T;
        std::shared_future<T> f0;
        std::shared_future<T> f;
        f = std::move(f0);
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef void T;
        std::promise<T> p;
        std::shared_future<T> f0 = p.get_future();
        std::shared_future<T> f;
        f = std::move(f0);
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef void T;
        std::shared_future<T> f0;
        std::shared_future<T> f;
        f = std::move(f0);
        assert(!f0.valid());
        assert(!f.valid());
    }

  return 0;
}
