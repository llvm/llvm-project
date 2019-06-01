//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class Y> shared_ptr(const shared_ptr<Y>& r, T *p);

#include <memory>
#include <cassert>

#include "test_macros.h"

struct B
{
    static int count;

    B() {++count;}
    B(const B&) {++count;}
    ~B() {--count;}
};

int B::count = 0;

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

int main(int, char**)
{
    {
        std::shared_ptr<A> pA(new A);
        assert(pA.use_count() == 1);
        {
            B b;
            std::shared_ptr<B> pB(pA, &b);
            assert(A::count == 1);
            assert(B::count == 1);
            assert(pA.use_count() == 2);
            assert(pB.use_count() == 2);
            assert(pB.get() == &b);
        }
        assert(pA.use_count() == 1);
        assert(A::count == 1);
        assert(B::count == 0);
    }
    assert(A::count == 0);
    assert(B::count == 0);

  return 0;
}
