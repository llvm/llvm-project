//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-exceptions
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: c++03

// <future>

// class shared_future<R>

// ~shared_future();

#include <future>
#include <cassert>

#include "test_macros.h"
#include "test_allocator.h"

int main(int, char**)
{
    test_allocator_statistics alloc_stats;
    assert(alloc_stats.alloc_count == 0);
    {
        typedef int T;
        std::shared_future<T> f;
        {
            std::promise<T> p(std::allocator_arg, test_allocator<T>(&alloc_stats));
            assert(alloc_stats.alloc_count == 1);
            f = p.get_future();
            assert(alloc_stats.alloc_count == 1);
            assert(f.valid());
        }
        assert(alloc_stats.alloc_count == 1);
        assert(f.valid());
    }
    assert(alloc_stats.alloc_count == 0);
    {
        typedef int& T;
        std::shared_future<T> f;
        {
            std::promise<T> p(std::allocator_arg, test_allocator<int>(&alloc_stats));
            assert(alloc_stats.alloc_count == 1);
            f = p.get_future();
            assert(alloc_stats.alloc_count == 1);
            assert(f.valid());
        }
        assert(alloc_stats.alloc_count == 1);
        assert(f.valid());
    }
    assert(alloc_stats.alloc_count == 0);
    {
        typedef void T;
        std::shared_future<T> f;
        {
            std::promise<T> p(std::allocator_arg, test_allocator<T>(&alloc_stats));
            assert(alloc_stats.alloc_count == 1);
            f = p.get_future();
            assert(alloc_stats.alloc_count == 1);
            assert(f.valid());
        }
        assert(alloc_stats.alloc_count == 1);
        assert(f.valid());
    }
    assert(alloc_stats.alloc_count == 0);

  return 0;
}
