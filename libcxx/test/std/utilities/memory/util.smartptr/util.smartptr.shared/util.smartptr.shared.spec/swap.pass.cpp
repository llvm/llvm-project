//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// shared_ptr

// template<class T> void swap(shared_ptr<T>& a, shared_ptr<T>& b);

#include <memory>
#include <cassert>

#include "test_macros.h"

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
        A* ptr1 = new A;
        A* ptr2 = new A;
        std::shared_ptr<A> p1(ptr1);
        {
            std::shared_ptr<A> p2(ptr2);
            swap(p1, p2);
            assert(p1.use_count() == 1);
            assert(p1.get() == ptr2);
            assert(p2.use_count() == 1);
            assert(p2.get() == ptr1);
            assert(A::count == 2);
        }
        assert(p1.use_count() == 1);
        assert(p1.get() == ptr2);
        assert(A::count == 1);
    }
    assert(A::count == 0);
    {
        A* ptr1 = new A;
        A* ptr2 = 0;
        std::shared_ptr<A> p1(ptr1);
        {
            std::shared_ptr<A> p2;
            swap(p1, p2);
            assert(p1.use_count() == 0);
            assert(p1.get() == ptr2);
            assert(p2.use_count() == 1);
            assert(p2.get() == ptr1);
            assert(A::count == 1);
        }
        assert(p1.use_count() == 0);
        assert(p1.get() == ptr2);
        assert(A::count == 0);
    }
    assert(A::count == 0);
    {
        A* ptr1 = 0;
        A* ptr2 = new A;
        std::shared_ptr<A> p1;
        {
            std::shared_ptr<A> p2(ptr2);
            swap(p1, p2);
            assert(p1.use_count() == 1);
            assert(p1.get() == ptr2);
            assert(p2.use_count() == 0);
            assert(p2.get() == ptr1);
            assert(A::count == 1);
        }
        assert(p1.use_count() == 1);
        assert(p1.get() == ptr2);
        assert(A::count == 1);
    }
    assert(A::count == 0);
    {
        A* ptr1 = 0;
        A* ptr2 = 0;
        std::shared_ptr<A> p1;
        {
            std::shared_ptr<A> p2;
            swap(p1, p2);
            assert(p1.use_count() == 0);
            assert(p1.get() == ptr2);
            assert(p2.use_count() == 0);
            assert(p2.get() == ptr1);
            assert(A::count == 0);
        }
        assert(p1.use_count() == 0);
        assert(p1.get() == ptr2);
        assert(A::count == 0);
    }
    assert(A::count == 0);

  return 0;
}
