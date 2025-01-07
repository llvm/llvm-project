//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class D, class A> shared_ptr(nullptr_t, D d, A a);

#include <memory>
#include <cassert>
#include "test_macros.h"
#include "deleter_types.h"
#include "test_allocator.h"
#include "min_allocator.h"

#include "types.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

// LWG 3233. Broken requirements for shared_ptr converting constructors
// https://cplusplus.github.io/LWG/issue3233
static_assert( std::is_constructible<std::shared_ptr<int>,  std::nullptr_t, test_deleter<int>, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int>,  std::nullptr_t, bad_deleter, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int>,  std::nullptr_t, no_nullptr_deleter, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int>,  std::nullptr_t, no_move_deleter, test_allocator<int> >::value, "");

#if TEST_STD_VER >= 17
static_assert( std::is_constructible<std::shared_ptr<int[]>,  std::nullptr_t, test_deleter<int>, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[]>,  std::nullptr_t, bad_deleter, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[]>,  std::nullptr_t, no_nullptr_deleter, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[]>,  std::nullptr_t, no_move_deleter, test_allocator<int> >::value, "");

static_assert( std::is_constructible<std::shared_ptr<int[5]>,  std::nullptr_t, test_deleter<int>, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[5]>,  std::nullptr_t, bad_deleter, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[5]>,  std::nullptr_t, no_nullptr_deleter, test_allocator<int> >::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[5]>,  std::nullptr_t, no_move_deleter, test_allocator<int> >::value, "");
#endif

int main(int, char**)
{
    test_allocator_statistics alloc_stats;
    {
        std::shared_ptr<A> p(nullptr, test_deleter<A>(3), test_allocator<A>(5, &alloc_stats));
        assert(A::count == 0);
        assert(p.use_count() == 1);
        assert(p.get() == 0);
        assert(test_deleter<A>::count == 1);
        assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
        test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
        assert(d);
        assert(d->state() == 3);
#endif
        assert(alloc_stats.count == 1);
        assert(alloc_stats.alloc_count == 1);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    assert(alloc_stats.count == 0);
    assert(alloc_stats.alloc_count == 0);
    test_deleter<A>::dealloc_count = 0;

    // Test an allocator with a minimal interface
    {
        std::shared_ptr<A> p(nullptr, test_deleter<A>(1), bare_allocator<void>());
        assert(A::count == 0);
        assert(p.use_count() == 1);
        assert(p.get() == 0);
        assert(test_deleter<A>::count ==1);
        assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
        test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
        assert(d);
        assert(d->state() == 1);
#endif
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    test_deleter<A>::dealloc_count = 0;

#if TEST_STD_VER >= 11
    // Test an allocator that returns class-type pointers
    {
        std::shared_ptr<A> p(nullptr, test_deleter<A>(1), min_allocator<void>());
        assert(A::count == 0);
        assert(p.use_count() == 1);
        assert(p.get() == 0);
        assert(test_deleter<A>::count ==1);
        assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
        test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
        assert(d);
        assert(d->state() == 1);
#endif
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
#endif

    // Make sure we can use this constructor with a pointer-to-const
    {
        std::shared_ptr<A const> p(nullptr, test_deleter<A const>(3), test_allocator<A>(5, &alloc_stats));
        (void)p;
    }

  return 0;
}
