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

// class future<R>

// ~future();

#include <future>
#include <cassert>

#include "test_allocator.h"

int main(int, char**)
{
    assert(test_alloc_base::alloc_count == 0);
    {
        typedef int T;
        std::future<T> f;
        {
            std::promise<T> p(std::allocator_arg, test_allocator<T>());
            assert(test_alloc_base::alloc_count == 1);
            f = p.get_future();
            assert(test_alloc_base::alloc_count == 1);
            assert(f.valid());
        }
        assert(test_alloc_base::alloc_count == 1);
        assert(f.valid());
    }
    assert(test_alloc_base::alloc_count == 0);
    {
        typedef int& T;
        std::future<T> f;
        {
            std::promise<T> p(std::allocator_arg, test_allocator<int>());
            assert(test_alloc_base::alloc_count == 1);
            f = p.get_future();
            assert(test_alloc_base::alloc_count == 1);
            assert(f.valid());
        }
        assert(test_alloc_base::alloc_count == 1);
        assert(f.valid());
    }
    assert(test_alloc_base::alloc_count == 0);
    {
        typedef void T;
        std::future<T> f;
        {
            std::promise<T> p(std::allocator_arg, test_allocator<T>());
            assert(test_alloc_base::alloc_count == 1);
            f = p.get_future();
            assert(test_alloc_base::alloc_count == 1);
            assert(f.valid());
        }
        assert(test_alloc_base::alloc_count == 1);
        assert(f.valid());
    }
    assert(test_alloc_base::alloc_count == 0);

  return 0;
}
