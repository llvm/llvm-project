//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// <future>

// class promise<R>

// template <class Allocator>
//   promise(allocator_arg_t, const Allocator& a);

#include <future>
#include <cassert>

#include "../test_allocator.h"

int main()
{
    assert(test_alloc_base::count == 0);
    {
        std::promise<int> p(std::allocator_arg, test_allocator<int>());
        assert(test_alloc_base::count == 1);
        std::future<int> f = p.get_future();
        assert(test_alloc_base::count == 1);
        assert(f.valid());
    }
    assert(test_alloc_base::count == 0);
    {
        std::promise<int&> p(std::allocator_arg, test_allocator<int>());
        assert(test_alloc_base::count == 1);
        std::future<int&> f = p.get_future();
        assert(test_alloc_base::count == 1);
        assert(f.valid());
    }
    assert(test_alloc_base::count == 0);
    {
        std::promise<void> p(std::allocator_arg, test_allocator<void>());
        assert(test_alloc_base::count == 1);
        std::future<void> f = p.get_future();
        assert(test_alloc_base::count == 1);
        assert(f.valid());
    }
    assert(test_alloc_base::count == 0);
}
