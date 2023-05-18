//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// weak_ptr

// template<class Y> weak_ptr& operator=(const shared_ptr<Y>& r);

#include <memory>
#include <type_traits>
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
    {
        const std::shared_ptr<A> pA(new A);
        assert(pA.use_count() == 1);
        {
            std::weak_ptr<B> pB;
            pB = pA;
            assert(B::count == 1);
            assert(A::count == 1);
            assert(pB.use_count() == 1);
            assert(pA.use_count() == 1);
        }
        assert(pA.use_count() == 1);
        assert(B::count == 1);
        assert(A::count == 1);
    }
    assert(B::count == 0);
    assert(A::count == 0);

#if TEST_STD_VER > 14
    {
        const std::shared_ptr<A[]> p1(new A[8]);
        assert(p1.use_count() == 1);
        {
            std::weak_ptr<const A[]> p2;
            p2 = p1;
            assert(A::count == 8);
            assert(p2.use_count() == 1);
            assert(p1.use_count() == 1);
        }
        assert(p1.use_count() == 1);
        assert(A::count == 8);
    }
    assert(A::count == 0);
#endif

  return 0;
}
