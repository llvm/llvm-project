//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// XFAIL: availability-aligned_allocation-missing

// <experimental/memory_resource>

// memory_resource * null_memory_resource()

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

#include <experimental/memory_resource>
#include <new>
#include <type_traits>
#include <cassert>

#include "test_macros.h"
#include "count_new.h"

namespace ex = std::experimental::pmr;

struct assert_on_compare : public ex::memory_resource
{
protected:
    void * do_allocate(std::size_t, size_t) override
    { assert(false); return nullptr; }

    void do_deallocate(void *, std::size_t, size_t) override
    { assert(false); }

    bool do_is_equal(ex::memory_resource const &) const noexcept override
    { assert(false); return false; }
};

void test_return()
{
    {
        static_assert(std::is_same<
            decltype(ex::null_memory_resource()), ex::memory_resource*
          >::value, "");
    }
    // Test that the returned value is not null
    {
        assert(ex::null_memory_resource());
    }
    // Test the same value is returned by repeated calls.
    {
        assert(ex::null_memory_resource() == ex::null_memory_resource());
    }
}

void test_equality()
{
    // Same object
    {
        ex::memory_resource & r1 = *ex::null_memory_resource();
        ex::memory_resource & r2 = *ex::null_memory_resource();
        // check both calls returned the same object
        assert(&r1 == &r2);
        // check for proper equality semantics
        assert(r1 == r2);
        assert(r2 == r1);
        assert(!(r1 != r2));
        assert(!(r2 != r1));
        // check the is_equal method
        assert(r1.is_equal(r2));
        assert(r2.is_equal(r1));
    }
    // Different types
    {
        ex::memory_resource & r1 = *ex::null_memory_resource();
        assert_on_compare c;
        ex::memory_resource & r2 = c;
        assert(r1 != r2);
        assert(!(r1 == r2));
        assert(!r1.is_equal(r2));
    }
}

void test_allocate()
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    DisableAllocationGuard g; // null_memory_resource shouldn't allocate.
    try {
        ex::null_memory_resource()->allocate(1);
        assert(false);
    } catch (std::bad_alloc const &) {
       // do nothing
    } catch (...) {
        assert(false);
    }
#endif
}

void test_deallocate()
{
    globalMemCounter.reset();

    int x = 42;
    ex::null_memory_resource()->deallocate(nullptr, 0);
    ex::null_memory_resource()->deallocate(&x, 0);

    assert(globalMemCounter.checkDeleteCalledEq(0));
    assert(globalMemCounter.checkDeleteArrayCalledEq(0));
}

int main(int, char**)
{
    test_return();
    test_equality();
    test_allocate();
    test_deallocate();

  return 0;
}
