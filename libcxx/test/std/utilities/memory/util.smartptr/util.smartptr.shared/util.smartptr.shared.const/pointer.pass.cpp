//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <memory>

// template<class Y> explicit shared_ptr(Y* p);

#include <cassert>
#include <memory>
#include <type_traits>

#include "test_macros.h"

struct A
{
    static int count;

    A() {++count;}
    A(const A&) {++count;}
    ~A() {--count;}
};

int A::count = 0;

struct Derived : A {};

// https://llvm.org/PR60258
// Invalid constructor SFINAE for std::shared_ptr's array ctors
static_assert( std::is_constructible<std::shared_ptr<int>,  int*>::value, "");
static_assert( std::is_constructible<std::shared_ptr<A>,  Derived*>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<A>,  int*>::value, "");

#if TEST_STD_VER >= 17
static_assert( std::is_constructible<std::shared_ptr<int[]>,  int*>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[]>,  int(*)[]>::value, "");
static_assert( std::is_constructible<std::shared_ptr<int[5]>, int*>::value, "");
static_assert(!std::is_constructible<std::shared_ptr<int[5]>, int(*)[5]>::value, "");
#endif

// Test explicit
static_assert(std::is_constructible<std::shared_ptr<int>, int*>::value, "");
static_assert(!std::is_convertible<int*, std::shared_ptr<int> >::value, "");

int main(int, char**)
{
    {
        assert(A::count == 0);
        A* ptr = new A;
        std::shared_ptr<A> p(ptr);
        assert(A::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
    }

    {
        assert(A::count == 0);
        A const* ptr = new A;
        std::shared_ptr<A const> p(ptr);
        assert(A::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
    }

    {
        assert(A::count == 0);
        A* ptr = new A;
        std::shared_ptr<void> p(ptr);
        assert(A::count == 1);
        assert(p.use_count() == 1);
        assert(p.get() == ptr);
    }

#if TEST_STD_VER > 14
    {
        assert(A::count == 0);
        std::shared_ptr<A[8]> pA(new A[8]);
        assert(pA.use_count() == 1);
        assert(A::count == 8);
    }

    {
        assert(A::count == 0);
        std::shared_ptr<A[]> pA(new A[8]);
        assert(pA.use_count() == 1);
        assert(A::count == 8);
    }

    {
        assert(A::count == 0);
        std::shared_ptr<const A[]> pA(new A[8]);
        assert(pA.use_count() == 1);
        assert(A::count == 8);
    }

    {
        assert(A::count == 0);
        std::shared_ptr<A> pA(new Derived);
        assert(pA.use_count() == 1);
        assert(A::count == 1);
    }
#endif

    return 0;
}
