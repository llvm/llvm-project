//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y, class D, class A> void reset(Y* p, D d, A a);

#include <memory>
#include <cassert>
#include "test_macros.h"
#include "deleter_types.h"
#include "test_allocator.h"

struct B
{
    static int count;

    B() {++count;}
    B(const B&) {++count;}
    virtual ~B() {--count;}
};

int B::count = 0;

struct A
    : public B
{
    static int count;

    A() {++count;}
    A(const A& other) : B(other) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    {
        std::shared_ptr<B> p(new B);
        A* ptr = new A;
        p.reset(ptr, test_deleter<A>(3), test_allocator<A>(4));
        assert(A::count == 1);
        assert(B::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
        assert(test_deleter<A>::count == 1);
        assert(test_deleter<A>::dealloc_count == 0);
#ifndef TEST_HAS_NO_RTTI
        test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
        assert(d);
        assert(d->state() == 3);
#endif
        assert(test_allocator<A>::count == 1);
        assert(test_allocator<A>::alloc_count == 1);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    assert(test_allocator<A>::count == 0);
    assert(test_allocator<A>::alloc_count == 0);
    {
        std::shared_ptr<B> p;
        A* ptr = new A;
        p.reset(ptr, test_deleter<A>(3), test_allocator<A>(4));
        assert(A::count == 1);
        assert(B::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
        assert(test_deleter<A>::count == 1);
        assert(test_deleter<A>::dealloc_count == 1);
#ifndef TEST_HAS_NO_RTTI
        test_deleter<A>* d = std::get_deleter<test_deleter<A> >(p);
        assert(d);
        assert(d->state() == 3);
#endif
        assert(test_allocator<A>::count == 1);
        assert(test_allocator<A>::alloc_count == 1);
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 2);
    assert(test_allocator<A>::count == 0);
    assert(test_allocator<A>::alloc_count == 0);

#if TEST_STD_VER > 14
    {
        std::shared_ptr<const A[]> p;
        A* ptr = new A[8];
        p.reset(ptr, CDeleter<A[]>(), test_allocator<A[]>());
        assert(A::count == 8);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
    }
    assert(A::count == 0);
#endif

  return 0;
}
