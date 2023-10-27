//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y, class D> void reset(Y* p, D d);

#include <cassert>
#include <memory>
#include "deleter_types.h"
#include "reset_helper.h"
#include "test_macros.h"

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

struct bad_ty { };
struct bad_deleter
{
    void operator()(bad_ty) { }
};

struct Base { };
struct Derived : Base { };

static_assert( HasReset<std::shared_ptr<int>,  int*, test_deleter<int> >::value, "");
static_assert(!HasReset<std::shared_ptr<int>,  int*, bad_deleter>::value, "");
static_assert( HasReset<std::shared_ptr<Base>,  Derived*, test_deleter<Base> >::value, "");
static_assert(!HasReset<std::shared_ptr<A>,  int*, test_deleter<A> >::value, "");

#if TEST_STD_VER >= 17
static_assert( HasReset<std::shared_ptr<int[]>,  int*, test_deleter<int>>::value, "");
static_assert(!HasReset<std::shared_ptr<int[]>,  int*, bad_deleter>::value, "");
static_assert(!HasReset<std::shared_ptr<int[]>,  int(*)[], test_deleter<int>>::value, "");
static_assert( HasReset<std::shared_ptr<int[5]>, int*, test_deleter<int>>::value, "");
static_assert(!HasReset<std::shared_ptr<int[5]>, int*, bad_deleter>::value, "");
static_assert(!HasReset<std::shared_ptr<int[5]>, int(*)[5], test_deleter<int>>::value, "");
#endif

int main(int, char**)
{
    {
        std::shared_ptr<B> p(new B);
        A* ptr = new A;
        p.reset(ptr, test_deleter<A>(3));
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
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 1);
    {
        std::shared_ptr<B> p;
        A* ptr = new A;
        p.reset(ptr, test_deleter<A>(3));
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
    }
    assert(A::count == 0);
    assert(test_deleter<A>::count == 0);
    assert(test_deleter<A>::dealloc_count == 2);

#if TEST_STD_VER > 14
    {
        std::shared_ptr<const A[]> p;
        A* ptr = new A[8];
        p.reset(ptr, CDeleter<A[]>());
        assert(A::count == 8);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
    }
    assert(A::count == 0);
#endif

  return 0;
}
