//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <functional>

// template<Returnable R, class T, CopyConstructible... Args>
//   unspecified mem_fn(R (T::* pm)(Args...) const);

#include <functional>
#include <cassert>

#include "test_macros.h"

struct A
{
    TEST_CONSTEXPR char test0() const {return 'a';}
    TEST_CONSTEXPR char test1(int) const {return 'b';}
    TEST_CONSTEXPR char test2(int, double) const {return 'c';}
};

template <class F>
TEST_CONSTEXPR_CXX20 bool
test0(F f)
{
    {
    A a;
    assert(f(a) == 'a');
    A* ap = &a;
    assert(f(ap) == 'a');
    const A* cap = &a;
    assert(f(cap) == 'a');
    const F& cf = f;
    assert(cf(ap) == 'a');
    }
    return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool
test1(F f)
{
    {
    A a;
    assert(f(a, 1) == 'b');
    A* ap = &a;
    assert(f(ap, 2) == 'b');
    const A* cap = &a;
    assert(f(cap, 2) == 'b');
    const F& cf = f;
    assert(cf(ap, 2) == 'b');
    }
    return true;
}

template <class F>
TEST_CONSTEXPR_CXX20 bool
test2(F f)
{
    {
    A a;
    assert(f(a, 1, 2) == 'c');
    A* ap = &a;
    assert(f(ap, 2, 3.5) == 'c');
    const A* cap = &a;
    assert(f(cap, 2, 3.5) == 'c');
    const F& cf = f;
    assert(cf(ap, 2, 3.5) == 'c');
    }
    return true;
}

int main(int, char**)
{
    test0(std::mem_fn(&A::test0));
    test1(std::mem_fn(&A::test1));
    test2(std::mem_fn(&A::test2));

#if TEST_STD_VER >= 20
    static_assert(test0(std::mem_fn(&A::test0)));
    static_assert(test1(std::mem_fn(&A::test1)));
    static_assert(test2(std::mem_fn(&A::test2)));
#endif

    return 0;
}
