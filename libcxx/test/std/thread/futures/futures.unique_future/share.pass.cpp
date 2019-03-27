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

// class future<R>

// shared_future<R> share() &&;

#include <future>
#include <cassert>

int main(int, char**)
{
    {
        typedef int T;
        std::promise<T> p;
        std::future<T> f0 = p.get_future();
        static_assert( noexcept(f0.share()), "");
        std::shared_future<T> f = f0.share();
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef int T;
        std::future<T> f0;
        static_assert( noexcept(f0.share()), "");
        std::shared_future<T> f = f0.share();
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef int& T;
        std::promise<T> p;
        std::future<T> f0 = p.get_future();
        static_assert( noexcept(f0.share()), "");
        std::shared_future<T> f = f0.share();
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef int& T;
        std::future<T> f0;
        static_assert( noexcept(f0.share()), "");
        std::shared_future<T> f = f0.share();
        assert(!f0.valid());
        assert(!f.valid());
    }
    {
        typedef void T;
        std::promise<T> p;
        std::future<T> f0 = p.get_future();
        static_assert( noexcept(f0.share()), "");
        std::shared_future<T> f = f0.share();
        assert(!f0.valid());
        assert(f.valid());
    }
    {
        typedef void T;
        std::future<T> f0;
        static_assert( noexcept(f0.share()), "");
        std::shared_future<T> f = f0.share();
        assert(!f0.valid());
        assert(!f.valid());
    }

  return 0;
}
