//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class Y, class D, class A> shared_ptr(Y* p, D d, A a);

#include <memory>
#include <cassert>
#include "test_macros.h"
#include "deleter_types.h"
#include "test_allocator.h"
#include "min_allocator.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

struct Base { };
struct Derived : Base { };

int main(int, char**)
{
    {
    A* ptr = new A;
    std::shared_ptr<A> p(ptr, test_deleter<A>(3), test_allocator<A>(5));
    assert(A::count == 1);
    assert(p.use_count() == 1);
    assert(p.get() == ptr);
    test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
    assert(test_deleter<A>::count == 1);
    assert(test_deleter<A>::dealloc_count == 0);
    assert(d);
    assert(d->state() == 3);
    assert(test_allocator<A>::count == 1);
    assert(test_allocator<A>::alloc_count == 1);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    assert(test_allocator<A>::count == 0);
    assert(test_allocator<A>::alloc_count == 0);
    test_deleter<A>::dealloc_count = 0;
    // Test an allocator with a minimal interface
    {
    A* ptr = new A;
    std::shared_ptr<A> p(ptr, test_deleter<A>(3), bare_allocator<void>());
    assert(A::count == 1);
    assert(p.use_count() == 1);
    assert(p.get() == ptr);
    test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
    assert(test_deleter<A>::count == 1);
    assert(test_deleter<A>::dealloc_count == 0);
    assert(d);
    assert(d->state() == 3);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    test_deleter<A>::dealloc_count = 0;
#if TEST_STD_VER >= 11
    // Test an allocator that returns class-type pointers
    {
    A* ptr = new A;
    std::shared_ptr<A> p(ptr, test_deleter<A>(3), min_allocator<void>());
    assert(A::count == 1);
    assert(p.use_count() == 1);
    assert(p.get() == ptr);
    test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
    assert(test_deleter<A>::count == 1);
    assert(test_deleter<A>::dealloc_count == 0);
    assert(d);
    assert(d->state() == 3);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
#endif

    {
        // Make sure that we can construct a shared_ptr where the element type and pointer type
        // aren't "convertible" but are "compatible".
        static_assert(!std::is_constructible<std::shared_ptr<Derived[4]>,
                                             Base[4], test_deleter<Derived[4]>,
                                             test_allocator<Derived[4]> >::value, "");
    }

  return 0;
}
