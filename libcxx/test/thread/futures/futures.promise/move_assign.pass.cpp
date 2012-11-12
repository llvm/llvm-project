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

// promise& operator=(promise&& rhs);

#include <future>
#include <cassert>

#include "../test_allocator.h"

int main()
{
#ifndef _LIBCPP_HAS_NO_RVALUE_REFERENCES
    assert(test_alloc_base::count == 0);
    {
        std::promise<int> p0(std::allocator_arg, test_allocator<int>());
        std::promise<int> p(std::allocator_arg, test_allocator<int>());
        assert(test_alloc_base::count == 2);
        p = std::move(p0);
        assert(test_alloc_base::count == 1);
        std::future<int> f = p.get_future();
        assert(test_alloc_base::count == 1);
        assert(f.valid());
        try
        {
            f = p0.get_future();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::no_state));
        }
        assert(test_alloc_base::count == 1);
    }
    assert(test_alloc_base::count == 0);
    {
        std::promise<int&> p0(std::allocator_arg, test_allocator<int>());
        std::promise<int&> p(std::allocator_arg, test_allocator<int>());
        assert(test_alloc_base::count == 2);
        p = std::move(p0);
        assert(test_alloc_base::count == 1);
        std::future<int&> f = p.get_future();
        assert(test_alloc_base::count == 1);
        assert(f.valid());
        try
        {
            f = p0.get_future();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::no_state));
        }
        assert(test_alloc_base::count == 1);
    }
    assert(test_alloc_base::count == 0);
    {
        std::promise<void> p0(std::allocator_arg, test_allocator<void>());
        std::promise<void> p(std::allocator_arg, test_allocator<void>());
        assert(test_alloc_base::count == 2);
        p = std::move(p0);
        assert(test_alloc_base::count == 1);
        std::future<void> f = p.get_future();
        assert(test_alloc_base::count == 1);
        assert(f.valid());
        try
        {
            f = p0.get_future();
            assert(false);
        }
        catch (const std::future_error& e)
        {
            assert(e.code() == make_error_code(std::future_errc::no_state));
        }
        assert(test_alloc_base::count == 1);
    }
    assert(test_alloc_base::count == 0);
#endif  // _LIBCPP_HAS_NO_RVALUE_REFERENCES
}
