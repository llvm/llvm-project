//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y> explicit shared_ptr(const weak_ptr<Y>& r);

#include <memory>
#include <cassert>

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

int main(int, char**)
{
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::weak_ptr<A> wp;
        try
        {
            std::shared_ptr<A> sp(wp);
            assert(false);
        }
        catch (std::bad_weak_ptr&)
        {
        }
        assert(A::count == 0);
    }
#endif
    {
        std::shared_ptr<A> sp0(new A);
        std::weak_ptr<A> wp(sp0);
        std::shared_ptr<A> sp(wp);
        assert(sp.use_count() == 2);
        assert(sp.get() == sp0.get());
        assert(A::count == 1);
    }
    assert(A::count == 0);
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        std::shared_ptr<A> sp0(new A);
        std::weak_ptr<A> wp(sp0);
        sp0.reset();
        try
        {
            std::shared_ptr<A> sp(wp);
            assert(false);
        }
        catch (std::bad_weak_ptr&)
        {
        }
    }
    assert(A::count == 0);
#endif

#if TEST_STD_VER > 14
    {
        std::shared_ptr<A[]> sp0(new A[8]);
        std::weak_ptr<A[]> wp(sp0);
        std::shared_ptr<const A[]> sp(wp);
        assert(sp.use_count() == 2);
        assert(sp.get() == sp0.get());
        assert(A::count == 8);
    }
    assert(A::count == 0);
#endif

  return 0;
}
